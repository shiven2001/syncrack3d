"""Does the sampled key light actually reach the cracks?

Builds walls with the real unified recipe (no rendering) and casts shadow rays from
each crack token to points on each light's disk through the built scene. A ray counts
as blocked by the first object it hits, split into the room shell and everything else.
Each sample is weighted by the area-light form factor (cos at wall * cos at the disk /
distance^2), and a disk emits only on the side facing its target, which is the wall
centre, so a crack behind the disk's plane gets nothing from that sample.

    # -> outputs/light_audit/audit.jsonl, one line per wall (appends; finished seeds are skipped)
    PYTHONPATH=. .venv/bin/python tools/audit_light_occlusion.py build --seeds 0-99

    # tables from audit.jsonl, no Blender work
    .venv/bin/python tools/audit_light_occlusion.py report

    # render one close-up of chosen walls three ways: as built, with the shell casting no
    # shadow (the rig the sampler intended) and with the key light off
    # -> outputs/light_audit/compare/{tiles,compare.jsonl,compare.png}
    PYTHONPATH=. .venv/bin/python tools/audit_light_occlusion.py compare --seeds 4,25,39

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "outputs" / "light_audit"
AUDIT = OUT / "audit.jsonl"
DISK_SAMPLES = 32
LIFT_M = 0.01  # start rays off the wall, past the crack cards 2 mm in front of it


def _disk_points(location, target, diameter):
    """Sunflower points on the disk, which faces its target, with the disk's facing direction."""
    from mathutils import Vector

    centre = Vector(location)
    facing = (Vector(target) - centre).normalized()
    side = facing.orthogonal().normalized()
    other = facing.cross(side)
    points = []
    for k in range(DISK_SAMPLES):
        radius = 0.5 * diameter * math.sqrt((k + 0.5) / DISK_SAMPLES)
        angle = k * math.pi * (3.0 - math.sqrt(5.0))
        points.append(centre + side * (radius * math.cos(angle)) + other * (radius * math.sin(angle)))
    return points, facing


def _first_blocker(scene, depsgraph, origin, goal):
    """The first shadow-casting object between two points, or None."""
    direction = goal - origin
    length = direction.length
    direction = direction.normalized()
    start = origin.copy()
    for _ in range(16):
        left = length - (start - origin).length
        if left <= 1e-4:
            return None
        hit, location, _normal, _index, obj, _matrix = scene.ray_cast(depsgraph, start, direction, distance=left)
        if not hit:
            return None
        if obj is not None and not obj.visible_shadow:
            start = location + direction * 1e-4
            continue
        return obj
    return None


def _light_at(scene, depsgraph, spec, wall_centre, point) -> dict:
    """What one light delivers to one wall point, before and after occlusion."""
    from mathutils import Vector

    samples, facing = _disk_points(spec["location"], spec.get("target") or wall_centre, spec["size_m"])
    origin = Vector(point) + Vector((0.0, -LIFT_M, 0.0))
    normal = Vector((0.0, -1.0, 0.0))
    total = shell = other = behind = 0.0
    weighted_incidence = clear_total = 0.0
    blockers = {}
    for sample in samples:
        to_light = sample - Vector(point)
        distance = to_light.length
        unit = to_light / distance
        at_wall = max(unit.dot(normal), 0.0)
        at_disk = (-unit).dot(facing)
        if at_disk <= 0.0:  # the disk emits away from its target only
            behind += 1
            continue
        weight = at_wall * at_disk / distance**2
        total += weight
        blocker = _first_blocker(scene, depsgraph, origin, sample)
        if blocker is None:
            clear_total += weight
            weighted_incidence += weight * math.degrees(math.acos(min(at_wall, 1.0)))
        elif blocker.name.startswith("CWG_Shell"):
            shell += weight
            blockers[blocker.name] = blockers.get(blocker.name, 0) + 1
        else:
            other += weight
            blockers[blocker.name] = blockers.get(blocker.name, 0) + 1
    return dict(
        unoccluded=total, clear=clear_total, shell=shell, other=other, behind_disk=behind / len(samples),
        incidence_at_point=weighted_incidence / clear_total if clear_total else None, blockers=blockers,
    )


def _camera_blocked(scene, depsgraph, location, target):
    from mathutils import Vector

    return _first_blocker(scene, depsgraph, Vector(location), Vector(target) + Vector((0.0, -LIFT_M, 0.0)))


def audit_wall(record: dict) -> dict:
    import bpy

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    spec = record["spec"]
    wall_centre = (0.0, 0.0, spec["height_m"] * 0.5)
    tokens = [dict(index=d["index"], u=d["u_m"], v=d["v_m"]) for d in record.get("defects_placed", [])]
    lights = []
    for light in record["lighting"]["lights"]:
        at_tokens = [_light_at(scene, depsgraph, light, wall_centre, (t["u"], 0.0, t["v"])) for t in tokens]
        lights.append(dict(role=light["role"], regime=light["regime"], incidence_deg=light["incidence_deg"],
                           azimuth_deg=light["azimuth_deg"], power_w=light["power_w"], size_m=light["size_m"],
                           location=light["location"], kelvin=light["kelvin"], at_tokens=at_tokens,
                           at_centre=_light_at(scene, depsgraph, light, wall_centre, wall_centre)))
    cameras = []
    for pose in record["cameras"]:
        obj = bpy.data.objects.get(pose["object_name"])
        if obj is None:
            continue
        forward = obj.matrix_world.to_3x3() @ __import__("mathutils").Vector((0.0, 0.0, -1.0))
        origin = obj.matrix_world.translation
        hit = origin + forward * (-origin.y / forward.y) if forward.y > 1e-6 else None
        blocker = _camera_blocked(scene, depsgraph, origin, hit) if hit is not None else None
        cameras.append(dict(role=pose["role"], blocker=None if blocker is None else blocker.name,
                            location=list(origin)))
    shell = record.get("room_shell") or {}
    return dict(seed=record["seed"], width_m=spec["width_m"], height_m=spec["height_m"], world=record["lighting"]["world_strength"],
                shell_depth_m=shell.get("depth_m"), has_ceiling=shell.get("has_ceiling"), returns=shell.get("returns"),
                tokens=len(tokens), lights=lights, cameras=cameras)


def build(seeds: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if AUDIT.exists():
        done = {json.loads(line)["seed"] for line in AUDIT.read_text().splitlines()}
    low, high = (int(x) for x in seeds.split("-"))
    todo = [seed for seed in range(low, high + 1) if seed not in done]
    if not todo:
        print("nothing to do")
        return

    import syncrack3d.build_dataset as bd

    sys.argv = ["build_dataset", "--recipe", "unified", "--seeds", ",".join(str(s) for s in todo),
                "--output", str(OUT / "scenes"), "--no-render"]
    real = bd.build_one

    def wrap(seed, args, out_root):
        record = real(seed, args, out_root)
        with AUDIT.open("a") as handle:
            handle.write(json.dumps(audit_wall(record)) + "\n")
        print(f"AUDITED {seed}", flush=True)
        return record

    bd.build_one = wrap
    bd.main()


def report() -> None:
    walls = [json.loads(line) for line in AUDIT.read_text().splitlines()]
    print(f"walls {len(walls)}; closed rooms {sum(bool(w['has_ceiling']) and len(w['returns'] or []) == 2 for w in walls)}; "
          f"with ceiling {sum(bool(w['has_ceiling']) for w in walls)}; cracks {sum(w['tokens'] for w in walls)}")

    rows = {}
    for wall in walls:
        key = next(light for light in wall["lights"] if light["role"] == "key")
        rows.setdefault(key["regime"], []).append((wall, key))

    def fraction(stat, part):
        return stat[part] / stat["unoccluded"] if stat["unoccluded"] > 0 else 0.0

    print("\nKEY LIGHT AT THE CRACKS, per crack, form-factor weighted")
    print(f"{'regime':16}{'walls':>6}{'cracks':>7}{'mean reach':>11}{'shell':>7}{'other':>7}{'<10% reach':>11}{'>90% reach':>11}"
          f"{'sampled alpha':>14}{'alpha at crack':>15}")
    every = []
    for regime in ("window_raking", "window_soft", "lamp_mixed", "ceiling_diffuse", "flash"):
        stats = [(key, stat) for _, key in rows.get(regime, []) for stat in key["at_tokens"]]
        if not stats:
            continue
        every += stats
        reach = [fraction(s, "clear") for _, s in stats]
        at_crack = [s["incidence_at_point"] for _, s in stats if s["incidence_at_point"] is not None]
        sampled = [k["incidence_deg"] for k, _ in stats]
        print(f"{regime:16}{len(rows[regime]):>6}{len(stats):>7}{statistics.mean(reach):>11.2f}"
              f"{statistics.mean(fraction(s, 'shell') for _, s in stats):>7.2f}"
              f"{statistics.mean(fraction(s, 'other') for _, s in stats):>7.2f}"
              f"{sum(r < .1 for r in reach) / len(reach):>11.2f}{sum(r > .9 for r in reach) / len(reach):>11.2f}"
              f"{statistics.median(sampled):>14.1f}{statistics.median(at_crack) if at_crack else float('nan'):>15.1f}")
    reach = [fraction(s, "clear") for _, s in every]
    print(f"{'ALL':16}{len(walls):>6}{len(every):>7}{statistics.mean(reach):>11.2f}"
          f"{statistics.mean(fraction(s, 'shell') for _, s in every):>7.2f}"
          f"{statistics.mean(fraction(s, 'other') for _, s in every):>7.2f}"
          f"{sum(r < .1 for r in reach) / len(reach):>11.2f}{sum(r > .9 for r in reach) / len(reach):>11.2f}")

    print("\nAT THE WALL CENTRE, the point every light is aimed at")
    for regime in ("window_raking", "window_soft", "lamp_mixed", "ceiling_diffuse", "flash"):
        keys = [key for _, key in rows.get(regime, [])]
        if keys:
            reach = [fraction(k["at_centre"], "clear") for k in keys]
            print(f"{regime:16} mean reach {statistics.mean(reach):.2f}  blocked outright (<10%) {sum(r < .1 for r in reach) / len(reach):.2f}")

    # Which incidence does a crack really get from a light that reaches it?
    bands = ((0, 15, "head-on"), (15, 35, "low"), (35, 62, "mid"), (62, 90, "raking"))
    print("\nEFFECTIVE INCIDENCE AT CRACKS with at least half the key's light arriving")
    lit = [s["incidence_at_point"] for _, s in every if fraction(s, "clear") >= .5 and s["incidence_at_point"] is not None]
    for low, high, name in bands:
        print(f"  {name:8} {low:>2}-{high:<2} deg: {sum(low <= a < high for a in lit) / max(len(lit), 1):.2f}  of the {len(lit)} lit cracks; "
              f"{sum(low <= a < high for a in lit) / len(every):.2f} of all cracks")
    print(f"  no usable key light: {1 - len(lit) / len(every):.2f} of all cracks")
    raking_lit = [s for k, s in every if k["regime"] == "window_raking" and fraction(s, "clear") >= .5]
    print(f"  cracks with a raking-window key that actually arrives: {len(raking_lit)} = {len(raking_lit) / len(every):.2f} of all cracks")

    fills = [light for wall in walls for light in wall["lights"] if light["role"] == "fill"]
    fill_stats = [s for light in fills for s in light["at_tokens"]]
    if fill_stats:
        print(f"\nFILL LIGHT: {len(fills)} fills, mean reach at cracks {statistics.mean(fraction(s, 'clear') for s in fill_stats):.2f}")
    no_key_or_fill = 0
    total = 0
    for wall in walls:
        key = next(light for light in wall["lights"] if light["role"] == "key")
        fill = next((light for light in wall["lights"] if light["role"] == "fill"), None)
        for i, stat in enumerate(key["at_tokens"]):
            total += 1
            fill_ok = fill is not None and fraction(fill["at_tokens"][i], "clear") >= .5
            no_key_or_fill += fraction(stat, "clear") < .1 and not fill_ok
    print(f"Cracks with neither key (>=10%) nor fill (>=50%): {no_key_or_fill / max(total, 1):.2f}, left to the world light")

    cams = [c for wall in walls for c in wall["cameras"]]
    for role in sorted({c["role"] for c in cams}):
        chosen = [c for c in cams if c["role"] == role]
        shell = sum(bool(c["blocker"]) and c["blocker"].startswith("CWG_Shell") for c in chosen)
        other = sum(bool(c["blocker"]) and not c["blocker"].startswith("CWG_Shell") for c in chosen)
        print(f"Cameras {role}: {len(chosen)}, view axis blocked by shell {shell / len(chosen):.3f}, by other {other / len(chosen):.3f}")


COMPARE = OUT / "compare"
VARIANTS = ("built", "no_shell_shadow", "key_off")


def _reach(stat) -> float:
    return stat["clear"] / stat["unoccluded"] if stat["unoccluded"] > 0 else 0.0


def _render_variant(stem: str, ev: float, tiles: Path | None = None) -> dict:
    import bpy
    import numpy as np

    from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler
    from cwg_infinigen.core.rendering.sensor_response import apply_sensor_response

    tiles = tiles or COMPARE / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.view_settings.exposure = 0.0
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    rgb = apply_sensor_response(NodeWrangler(tree), layers.outputs["Image"], enabled=True, auto_exposure=True,
                                ev_compensation=ev, noise=False)

    def output(name, socket, kind, mode):
        node = tree.nodes.new("CompositorNodeOutputFile")
        node.base_path = str(tiles)
        node.format.file_format, node.format.color_mode = kind, mode
        if kind == "OPEN_EXR":
            node.format.color_depth = "32"
        node.file_slots[0].path = name
        tree.links.new(socket, node.inputs[0])

    output(f"{stem}_rgb_", rgb, "PNG", "RGB")
    output(f"{stem}_linear_", layers.outputs["Image"], "OPEN_EXR", "RGB")
    output(f"{stem}_label_", layers.outputs["DefectCrack"], "PNG", "BW")
    bpy.ops.render.render(write_still=False)

    def latest(prefix):
        return sorted(tiles.glob(f"{prefix}*"), key=lambda item: item.stat().st_mtime)[-1]

    image = bpy.data.images.load(str(latest(f"{stem}_linear_")))
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    linear = pixels.reshape(height, width, 4)[::-1, :, :3]
    return dict(rgb=str(latest(f"{stem}_rgb_")), label=str(latest(f"{stem}_label_")),
                linear_mean=float(linear.mean()), linear=linear)


def _metrics(rgb_path: str, label_path: str, linear) -> dict:
    """Crack visibility in the saved image, in the ring of wall around the crack (6-20 px)."""
    import cv2
    import numpy as np

    rgb = cv2.imread(rgb_path)
    lum = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(float)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    label = label.max(axis=2) if label.ndim == 3 else label
    mask = label >= 0.5 * label.max() if label.max() > 0 else label > 0
    m = mask.astype("uint8")
    ring = (cv2.dilate(m, np.ones((41, 41), "uint8")) > 0) & ~(cv2.dilate(m, np.ones((13, 13), "uint8")) > 0)
    if not mask.any() or not ring.any():
        return dict(crack_px=int(mask.sum()))
    wall = float(np.median(lum[ring]))
    texture = float(np.abs(lum[ring] - wall).mean())
    crack = float(np.abs(lum[mask] - wall).mean())
    linear_lum = linear.mean(axis=2)
    return dict(crack_px=int(mask.sum()), wall_level=wall, contrast=crack / max(wall, 1.0),
                crack_to_texture=crack / texture if texture > .5 else None,
                wall_linear=float(np.median(linear_lum[ring])))


def compare(seeds: str) -> None:
    import bpy

    COMPARE.mkdir(parents=True, exist_ok=True)
    audit = {wall["seed"]: wall for wall in (json.loads(line) for line in AUDIT.read_text().splitlines())}
    results = COMPARE / "compare.jsonl"

    import syncrack3d.build_dataset as bd
    from syncrack3d.render import configure_cycles

    sys.argv = ["build_dataset", "--recipe", "unified", "--seeds", seeds, "--output", str(COMPARE / "scenes"), "--no-render"]
    real = bd.build_one

    def wrap(seed, args, out_root):
        record = real(seed, args, out_root)
        key = next(light for light in audit[seed]["lights"] if light["role"] == "key")
        reaches = [_reach(stat) for stat in key["at_tokens"]]
        closeups = [pose for pose in record["cameras"] if pose["role"] == "closeup"
                    and bpy.data.objects.get(pose["object_name"]) is not None]
        # The close-up of the crack that this wall's reach is most typical of: its lowest for blocked walls.
        pose = min(closeups, key=lambda c: reaches[c["target_index"]] if statistics.mean(reaches) < .5 else -reaches[c["target_index"]])
        scene = bpy.context.scene
        scene.camera = bpy.data.objects[pose["object_name"]]
        scene.frame_set(1)
        configure_cycles(resolution=640, samples=64, denoise=True, dicing_rate=0.75)
        evs = record.get("camera_exposure_ev") or []
        ev = float(evs[record["cameras"].index(pose)]) if len(evs) == len(record["cameras"]) else 0.0
        shell = [obj for obj in bpy.data.objects if obj.name.startswith("CWG_Shell")]
        key_light = bpy.data.objects["CWG_Light_key"]
        row = dict(seed=seed, regime=key["regime"], incidence_deg=key["incidence_deg"], reach=reaches[pose["target_index"]],
                   camera=pose["object_name"], world=audit[seed]["world"], has_fill=len(audit[seed]["lights"]) > 1)
        for variant in VARIANTS:
            for obj in shell:
                obj.visible_shadow = variant != "no_shell_shadow"
            energy = key_light.data.energy
            if variant == "key_off":
                key_light.data.energy = 0.0
            shot = _render_variant(f"{seed:05d}_{variant}", ev)
            key_light.data.energy = energy
            row[variant] = dict(rgb=shot["rgb"], label=shot["label"], **_metrics(shot["rgb"], shot["label"], shot["linear"]))
        for obj in shell:
            obj.visible_shadow = True
        with results.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"COMPARED {seed} reach={row['reach']:.2f} " +
              " ".join(f"{v}: contrast={row[v].get('contrast', float('nan')):.3f} linear={row[v].get('wall_linear', float('nan')):.4f}" for v in VARIANTS), flush=True)
        return record

    bd.build_one = wrap
    bd.main()


def sheet() -> None:
    """One row per wall: the three variants side by side, as the saved 8-bit images."""
    import cv2
    import numpy as np

    rows = [json.loads(line) for line in (COMPARE / "compare.jsonl").read_text().splitlines()]
    tile, pad, head, gutter = 320, 6, 34, 210
    canvas = np.full((head + len(rows) * (tile + pad), gutter + 3 * (tile + pad), 3), 255, "uint8")
    font = cv2.FONT_HERSHEY_SIMPLEX
    for j, name in enumerate(("As built", "Shell casts no shadow", "Key light off")):
        cv2.putText(canvas, name, (gutter + j * (tile + pad) + 6, 24), font, .6, (0, 0, 0), 1, cv2.LINE_AA)
    for i, row in enumerate(rows):
        y = head + i * (tile + pad)
        lines = (f"seed {row['seed']}", row["regime"], f"alpha {row['incidence_deg']:.0f}", f"key reach {row['reach']:.2f}")
        for k, text in enumerate(lines):
            cv2.putText(canvas, text, (6, y + 24 + 24 * k), font, .55, (0, 0, 0), 1, cv2.LINE_AA)
        for j, variant in enumerate(VARIANTS):
            x = gutter + j * (tile + pad)
            image = cv2.resize(cv2.imread(row[variant]["rgb"]), (tile, tile), interpolation=cv2.INTER_AREA)
            canvas[y:y + tile, x:x + tile] = image
            value = row[variant].get("contrast")
            if value is not None:
                cv2.putText(canvas, f"contrast {value:.3f}", (x + 6, y + tile - 10), font, .5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, f"contrast {value:.3f}", (x + 6, y + tile - 10), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(COMPARE / "compare.png"), canvas)
    print(COMPARE / "compare.png")
    print(f"{'seed':>5} {'regime':16}{'reach':>6}  {'built':>7} {'no shell':>9} {'key off':>8}   {'light from key (built)':>22}  {'wall linear built / intended':>30}")
    for row in rows:
        built, free, off = (row[v] for v in VARIANTS)
        if "wall_linear" not in built:
            continue
        share = 1 - off["wall_linear"] / built["wall_linear"] if built["wall_linear"] > 0 else float("nan")
        print(f"{row['seed']:>5} {row['regime']:16}{row['reach']:>6.2f}  {built['contrast']:>7.3f} {free['contrast']:>9.3f} {off['contrast']:>8.3f}   "
              f"{share:>22.2f}  {built['wall_linear']:>14.4f} / {free['wall_linear']:<14.4f}")


PREVIEW = OUT / "preview"
OPTIONS = ("built", "inside_room", "shell_unlinked")
OPTION_TITLES = ("As built", "Option 1: light inside room", "Option 2: shell ignores lights")
ROOM_MARGIN_M = 0.10
WALL_GAP_M = 0.02


def fit_inside_room(light: dict, width, height, depth) -> dict:
    """Option 1: keep the direction from the target (alpha, beta), shorten r, and shrink the
    disk only as far as it needs to stay clear of the wall and the shell."""
    from mathutils import Vector

    target = Vector(light.get("target") or (0.0, 0.0, height * 0.5))
    offset = Vector(light["location"]) - target
    r, unit = offset.length, offset.normalized()
    limits = [r]
    if abs(unit.x) > 1e-6:
        limits.append((width * 0.5 - ROOM_MARGIN_M - (target.x if unit.x > 0 else -target.x)) / abs(unit.x))
    if abs(unit.z) > 1e-6:
        limits.append((height - ROOM_MARGIN_M - target.z) / unit.z if unit.z > 0 else (target.z - ROOM_MARGIN_M) / -unit.z)
    if abs(unit.y) > 1e-6:
        limits.append((depth - ROOM_MARGIN_M) / abs(unit.y))
    fitted = max(min(limits), 0.05)
    centre = target + unit * fitted
    radius = light["size_m"] * 0.5
    facing = -unit
    clear = (("x", width * 0.5 - abs(centre.x)), ("y", min(-centre.y - WALL_GAP_M, depth + centre.y)),
             ("z", min(centre.z, height - centre.z)))
    for axis, room in clear:
        spread = math.sqrt(max(1.0 - getattr(facing, axis) ** 2, 0.0))
        if spread > 1e-6:
            radius = min(radius, max(room, 0.0) / spread)
    radius = max(radius, 0.01)
    return dict(light, location=tuple(centre), size_m=radius * 2.0, r_sampled=r, r_fitted=fitted,
                target=tuple(target))


def preview(seeds: str) -> None:
    import bpy

    PREVIEW.mkdir(parents=True, exist_ok=True)
    audit = {wall["seed"]: wall for wall in (json.loads(line) for line in AUDIT.read_text().splitlines())}
    results = PREVIEW / "preview.jsonl"

    import syncrack3d.build_dataset as bd
    from syncrack3d.render import configure_cycles

    sys.argv = ["build_dataset", "--recipe", "unified", "--seeds", seeds, "--output", str(PREVIEW / "scenes"), "--no-render"]
    real = bd.build_one

    def wrap(seed, args, out_root):
        record = real(seed, args, out_root)
        scene = bpy.context.scene
        spec, wall_audit = record["spec"], audit[seed]
        width, height, depth = spec["width_m"], spec["height_m"], record["room_shell"]["depth_m"]
        wall_centre = (0.0, 0.0, height * 0.5)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        key = next(light for light in wall_audit["lights"] if light["role"] == "key")
        reaches = [_reach(stat) for stat in key["at_tokens"]]
        closeups = [c for c in record["cameras"] if c["role"] == "closeup" and bpy.data.objects.get(c["object_name"])]
        clear_context = {c["location"][0] for c in wall_audit["cameras"] if c["role"] == "context" and not c["blocker"]}
        context = [c for c in record["cameras"] if c["role"] == "context" and bpy.data.objects.get(c["object_name"])
                   and bpy.data.objects[c["object_name"]].matrix_world.translation.x in clear_context]
        views = [("closeup", min(closeups, key=lambda c: reaches[c["target_index"]]))]
        if context:
            views.append(("context", context[0]))

        # Option 2: only these objects can shadow the sampled lights, so the shell stops them
        # while the sky and bounced light still see it as solid.
        blockers = bpy.data.collections.new("preview_blockers")
        for obj in bpy.data.objects:
            if obj.type == "MESH" and not obj.name.startswith("CWG_Shell"):
                blockers.objects.link(obj)

        lights = {light["role"]: light for light in record["lighting"]["lights"]}
        fitted = {role: fit_inside_room(light, width, height, depth) for role, light in lights.items()}
        token_points = [(d["u_m"], 0.0, d["v_m"]) for d in record["defects_placed"]]
        row = dict(seed=seed, regime=key["regime"], incidence_deg=key["incidence_deg"], azimuth_deg=key["azimuth_deg"],
                   world=record["lighting"]["world_strength"], reach_built=statistics.mean(reaches) if reaches else None,
                   r_sampled=fitted["key"]["r_sampled"], r_fitted=fitted["key"]["r_fitted"],
                   size_sampled=lights["key"]["size_m"], size_fitted=fitted["key"]["size_m"],
                   reach_inside=statistics.mean(_reach(_light_at(scene, depsgraph, fitted["key"], wall_centre, point))
                                                for point in token_points) if token_points else None,
                   views={})
        configure_cycles(resolution=640, samples=64, denoise=True, dicing_rate=0.75)
        evs = record.get("camera_exposure_ev") or []
        for view, pose in views:
            scene.camera = bpy.data.objects[pose["object_name"]]
            scene.frame_set(1)
            ev = float(evs[record["cameras"].index(pose)]) if len(evs) == len(record["cameras"]) else 0.0
            row["views"][view] = {}
            for option in OPTIONS:
                originals = {}
                for role in lights:
                    obj = bpy.data.objects[f"CWG_Light_{role}"]
                    originals[role] = (obj.location.copy(), obj.data.size, obj.light_linking.blocker_collection)
                    if option == "inside_room":
                        obj.location, obj.data.size = fitted[role]["location"], fitted[role]["size_m"]
                    elif option == "shell_unlinked":
                        obj.light_linking.blocker_collection = blockers
                shot = _render_variant(f"{seed:05d}_{view}_{option}", ev, PREVIEW / "tiles")
                for role, (location, size, linked) in originals.items():
                    obj = bpy.data.objects[f"CWG_Light_{role}"]
                    obj.location, obj.data.size = location, size
                    obj.light_linking.blocker_collection = linked
                linear = shot.pop("linear")
                luminance = linear.mean(axis=2)
                import numpy as np

                low, high = np.percentile(luminance, (5, 95))
                metrics = _metrics(shot["rgb"], shot["label"], linear)
                import cv2

                saved = cv2.imread(shot["rgb"])
                metrics.update(rgb=shot["rgb"], falloff=float(high / max(low, 1e-6)),
                               clipped=float((saved.max(axis=2) >= 254).mean()))
                row["views"][view][option] = metrics
        with results.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"PREVIEWED {seed}", flush=True)
        return record

    bd.build_one = wrap
    bd.main()


def preview_sheets() -> None:
    import cv2
    import numpy as np

    rows = [json.loads(line) for line in (PREVIEW / "preview.jsonl").read_text().splitlines()]
    font = cv2.FONT_HERSHEY_SIMPLEX
    tile, pad, head, gutter = 460, 8, 44, 250
    for view in ("closeup", "context"):
        chosen = [row for row in rows if view in row["views"]]
        canvas = np.full((head + len(chosen) * (tile + pad), gutter + 3 * (tile + pad), 3), 255, "uint8")
        for j, title in enumerate(OPTION_TITLES):
            cv2.putText(canvas, title, (gutter + j * (tile + pad) + 6, 30), font, .68, (0, 0, 0), 2, cv2.LINE_AA)
        for i, row in enumerate(chosen):
            y = head + i * (tile + pad)
            lines = (f"seed {row['seed']}", row["regime"], f"alpha {row['incidence_deg']:.0f}, beta {row['azimuth_deg']:.0f}",
                     f"r {row['r_sampled']:.1f} -> {row['r_fitted']:.1f} m", f"disk {row['size_sampled']:.2f} -> {row['size_fitted']:.2f} m")
            for k, text in enumerate(lines):
                cv2.putText(canvas, text, (6, y + 30 + 28 * k), font, .55, (0, 0, 0), 1, cv2.LINE_AA)
            for j, option in enumerate(OPTIONS):
                x = gutter + j * (tile + pad)
                canvas[y:y + tile, x:x + tile] = cv2.resize(cv2.imread(row["views"][view][option]["rgb"]), (tile, tile),
                                                            interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(PREVIEW / f"preview_{view}.png"), canvas)
        print(PREVIEW / f"preview_{view}.png")
    print(f"{'seed':>5} {'regime':15}{'view':>8}{'option':>16}{'contrast':>9}{'falloff':>9}{'clipped':>9}{'wall linear':>12}")
    for row in rows:
        print(f"seed {row['seed']}: reach built {row['reach_built']:.2f}, option 1 {row['reach_inside']:.2f}")
        for view, options in row["views"].items():
            for option, m in options.items():
                print(f"{row['seed']:>5} {row['regime']:15}{view:>8}{option:>16}{m.get('contrast', float('nan')):>9.3f}"
                      f"{m['falloff']:>9.1f}{m['clipped']:>9.3f}{m.get('wall_linear', float('nan')):>12.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("build", "report", "compare", "sheet", "preview", "preview_sheets"))
    parser.add_argument("--seeds", default="0-99")
    args = parser.parse_args()
    if args.stage == "build":
        build(args.seeds)
    elif args.stage == "compare":
        compare(args.seeds)
    elif args.stage == "sheet":
        sheet()
    elif args.stage == "preview":
        preview(args.seeds)
    elif args.stage == "preview_sheets":
        preview_sheets()
    else:
        report()


if __name__ == "__main__":
    main()
