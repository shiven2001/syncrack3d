#!/usr/bin/env python3
"""SynCrack3D dataset driver. One seed is one wall, built in seven steps:

1. wall and material: a 15-category palette drawn uniformly (material_diversity);
   the wall draws a crack style (styles.py) that sets its crack morphology and
   setting (indoor room or outdoor facade);
2. context: an indoor room shell (ceiling, side walls in the wall's own finish or
   both in one other wall category, an Infinigen floor) or an outdoor facade with ground and sky;
3. fixtures: planned on their own stream and built (switched sockets, doors,
   trims, services; windows with sills and downpipes outdoors);
4. crack tokens around the fixtures (none under a large assembly, some beside a
   service run), a share snapped to fixture corners and edges;
5. crack injection into the wall material (displaced relief, coating edges, labels);
6. lighting and cameras (close-ups per crack plus context views; a camera jammed
   against an object is dropped);
7. one room's props (furniture against the wall, loose items on the floor), clear of
   cameras, lights and fixtures; then render RGB and crack AOVs.

    python syncrack3d/build_dataset.py --seeds 0-49 --output out/walls

Run it with the same Python that has bpy installed. Defaults are the SynCrack3D
v1.18.0 configuration; the remaining options are run controls and ablation
switches.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
import numpy as np  # noqa: E402

from syncrack3d.cameras import drop_blocked_cameras, sample_camera_poses  # noqa: E402
from syncrack3d.defects import (  # noqa: E402
    build_defects,
    inject_defects,
    sample_defect_plan,
)
from syncrack3d.framing import sample_exposure_evs  # noqa: E402
from syncrack3d.geometry import build_scene  # noqa: E402
from syncrack3d.materials import enable_true_displacement  # noqa: E402
from syncrack3d.lighting import (  # noqa: E402
    LIGHT_PROFILES,
    apply_lighting,
    sample_lighting_plan,
)
from syncrack3d.junctions import snap_to_junctions  # noqa: E402
from syncrack3d.styles import DEFAULT_MIX, parse_style_mix, wall_style  # noqa: E402
from syncrack3d.structures import (  # noqa: E402
    build_structures,
    crack_regions,
    sample_structure_plan,
)
from syncrack3d.occlusion import _world_bounds as prop_bounds  # noqa: E402
from syncrack3d.occlusion import sample_props  # noqa: E402
from syncrack3d.outdoor import (  # noqa: E402
    apply_sky,
    apply_weathering,
    build_outdoor,
    outdoor_lighting,
    sample_outdoor_plan,
)
from syncrack3d.room_shell import (  # noqa: E402
    apply_matched_finish,
    build_room_shell,
    sample_shell_plan,
)
from syncrack3d.render import (  # noqa: E402
    configure_cycles,
    render_cameras,
    setup_passes,
)
from syncrack3d.scene_setup import (  # noqa: E402
    CAPTURE_COLLECTION,
    configure_capture_scene,
)
from syncrack3d.spec import sample_scene_spec  # noqa: E402
from syncrack3d.material_diversity import (  # noqa: E402
    CATEGORIES as MATERIAL_CATEGORIES,
    apply_wall_material,
    is_clean_wall,
    virtual_targets,
    wall_relief,
)

logger = logging.getLogger("syncrack3d")


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def clear_scene():
    """Remove every object, including ones no operator can reach.

    Deliberately not select_all + delete. `hide_planes` hides each injected
    defect card, and a hidden object cannot be selected, so the operator route
    silently leaves every card from the previous seed in the file. They still
    carry their defect materials, so `classify_plane` finds them on the next
    run, they have no binding in the new scene's state, and injection rejects
    the whole scene with `unresolved_intended_host_relation`. Removing through
    bpy.data has no such blind spot.
    """
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)
    # Materials outlive their objects and injection keys its per-host registry
    # on material identity, so orphans here become stale hosts later.
    for block in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.lights,
                  bpy.data.cameras, bpy.data.node_groups, bpy.data.images):
        for datablock in list(block):
            if datablock.users == 0:
                block.remove(datablock)
    # Injection keeps a module-level registry of hosts it has already seen. It
    # holds references to objects from the previous scene, so not clearing it
    # between seeds injects scene N's defects into scene N-1's dead materials.
    from cwg_infinigen.assets.defect_injection import reset_registry

    reset_registry()


def build_one(seed: int, args, out_root: Path) -> dict:
    """Build (and render) one wall. `args` already carries the wall's crack style."""
    t_start, t_render = time.perf_counter(), None
    shell_finish = {"finish": "plain"} if args.shell_finish == "plain" else None
    scene_dir = out_root / f"seed_{seed:05d}"
    frames_dir = out_root / "frames" / f"seed_{seed:05d}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    spec = sample_scene_spec(seed, module="flat", substrate="paint")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    rng = random.Random(seed)

    clear_scene()
    build = build_scene(spec)
    wall = build.hosts[0]
    # The camera-facing plane is at y=0, with geometric normal -Y. Positive
    # height must move toward cameras at negative Y; no sign flip.
    wall["syndefect_displacement_sign"] = 1

    # 1. Wall material. The material helpers import Infinigen's room decorator
    # lazily, and that first import draws from numpy, so only a process's first
    # wall saw the extra draws. Import first, then reseed, so the wall is a
    # function of its own seed only.
    import cwg_infinigen.core.constraints.example_solver.room.decorate  # noqa: F401
    from cwg_infinigen.core import surface  # noqa: F401
    from cwg_infinigen.core.util.math import int_hash

    np.random.seed(int_hash((seed, "cwg_wall_material")) % (2**32))
    generator_name, surface_finish, material_record = apply_wall_material(
        wall, seed, args.material_category, args.material_categories)
    clean_wall = is_clean_wall(seed, material_record["category"], args.clean_wall_scale,
                               uniform=args.clean_wall_uniform)

    # 2. Context.
    shell_plan = sample_shell_plan(rng)  # drawn for every wall: keeps the shared RNG stream
    environment = args.environment
    outdoor_plan = sample_outdoor_plan(seed, spec.width_m) if environment == "outdoor" else None
    if outdoor_plan is None and args.structures != "off":
        # Fixture views reach wall boundaries; close the room so sockets and
        # door frames are not photographed against the sky outside the slab.
        shell_plan = replace(shell_plan, has_ceiling=True, returns=("left", "right"))
    if outdoor_plan is None:
        shell = build_room_shell(spec, shell_plan)
        if args.shell_finish == "matched":
            shell_finish = apply_matched_finish(shell, wall, seed, args.material_categories)
    else:
        build_outdoor(spec, outdoor_plan, wall)
    configure_capture_scene(spec, build.root)
    capture_collection = bpy.data.collections[CAPTURE_COLLECTION]

    # 3. Fixtures, on their own stream and blind to cracks, built now.
    structure_plans = sample_structure_plan(seed, spec, args.structures, environment)
    relief = round(wall_relief(wall, material_record["category"]), 5)
    structures = build_structures(structure_plans, relief, spec.height_m, spec.width_m)
    skirting = bpy.data.objects.get("CWG_Skirting")
    if skirting is not None:  # the base wall's own board, when no fixture plan replaced it
        skirting.location.y -= relief
    keepout, near = crack_regions(structure_plans)

    # 4. Crack tokens around the fixtures, then a share snapped to their corners and edges.
    plan = sample_defect_plan(
        rng,
        total_range=(0, 0) if clean_wall else (args.min_defects, args.max_defects),
        substrate_finish=surface_finish,
        size_range_m=tuple(args.defect_size_range),
    )
    if args.crack_profile == "network":
        # Network patches reach crack_network_extent x the token size; spaced only
        # 0.30 m apart they overlapped and blended instance ids where they met.
        plan = replace(plan, min_separation_m=max(plan.min_separation_m,
                                                  args.crack_network_extent * plan.size_range_m[1]))
    placements = build_defects(spec, wall, rng, plan, keepout=keepout, near=near,
                               near_fraction=args.crack_near_service_fraction)
    layout_record = {"order": "fixtures-first", "keepout_regions": len(keepout), "service_runs": len(near),
                     "tokens_requested": sum(plan.counts.values()), "tokens_placed": len(placements)}
    placements, junctions = snap_to_junctions(placements, structure_plans, spec, seed, args.crack_junction_fraction)

    # 5. Injection.
    manifest_path = scene_dir / "defect_instances.json"
    import gin
    import cwg_infinigen.assets.defect_injection  # noqa: F401

    with gin.unlock_config():
        if args.crack_render != "geometry":
            gin.bind_parameter("_inject_crack.crack_render", args.crack_render)
        if args.crack_layered_fraction > 0:
            gin.bind_parameter("_inject_crack.layered_fraction", args.crack_layered_fraction)
        gin.bind_parameter("_inject_crack.crack_profile", args.crack_profile)
        gin.bind_parameter("_inject_crack.crack_network_grime", tuple(args.crack_network_grime))
        gin.bind_parameter("_inject_crack.crack_network_extent", args.crack_network_extent)
        gin.bind_parameter("_inject_crack.crack_weathering_fraction", args.crack_weathering_fraction)
        gin.bind_parameter("_inject_crack.crack_width_scale", args.crack_width_scale)
        gin.bind_parameter("_inject_crack.crack_depth_scale", args.crack_depth_scale)
    injected = inject_defects(
        placements,
        wall,
        manifest_path=manifest_path,
        scene_id=f"cwg_seed_{seed:05d}",
        scene_seed=seed,
        dicing_rate=args.dicing_rate,
    )
    # After injection: a cracked wall's material slot is the private copy,
    # and a clean wall never enters injection at all. Both need real
    # displacement, or the clean negatives stay bump-only.
    enable_true_displacement(wall, args.dicing_rate)

    # 6. Lighting and cameras.
    lighting = sample_lighting_plan(rng, spec.width_m, spec.height_m)
    if args.light_profile != "default" and not args.relight_per_view:
        # Independent stream: walls, defects and cameras stay as in the default run.
        lighting = sample_lighting_plan(random.Random(seed ^ 0x4C494748), spec.width_m, spec.height_m,
                                        profile=args.light_profile)
    apply_lighting(lighting, spec.height_m, capture_collection)
    weathering = {}
    if outdoor_plan is not None:
        # One sun (or overcast sky) per wall, visible sky; the indoor plan above
        # was still drawn so the shared RNG stream is unchanged.
        lighting = outdoor_lighting(outdoor_plan, spec)
        apply_lighting(lighting, spec.height_m, capture_collection)
        apply_sky(outdoor_plan)
        weathering = apply_weathering(wall, outdoor_plan, spec.height_m)
    poses = sample_camera_poses(
        rng,
        spec,
        # a clean wall has no defects to frame: close-ups of plain wall instead
        virtual_targets(seed, spec) if clean_wall and not placements else placements,
        capture_collection,
        closeups_per_defect=args.closeups_per_defect,
        context_view_fraction=args.context_view_fraction,
        frame_fill_range=tuple(args.closeup_fill),
        max_yaw_deg=args.closeup_max_yaw,
        max_pitch_deg=args.closeup_max_pitch,
        preserve_closeup_scale=True,
        aim_jitter=args.closeup_aim_jitter,
    )
    poses, blocked = drop_blocked_cameras(poses, wall)
    layout_record["cameras_dropped"] = blocked

    # 7. Props last, one room's furniture around the cameras, lights, fixtures and
    # skirting. None outdoors. Views then jammed against a prop are dropped too.
    props, prop_room = [], None
    if outdoor_plan is None:
        def around(obj, r):
            at = np.array(obj.matrix_world.translation)
            return at - r, at + r

        obstacles = [around(bpy.data.objects[p.object_name], 0.15) for p in poses]
        obstacles += [around(o, 0.15) for o in bpy.data.objects if o.type == "LIGHT"]
        obstacles += [b for o in bpy.data.objects
                      if (o.get("cwg_role") == "structure" or o.get("cwg_hard_negative") == "skirting")
                      and o.type in ("MESH", "CURVE") and (b := prop_bounds(o)) is not None]
        prop_room, props = sample_props(rng, spec, placements, rooms=None, count_range=(0, args.max_props),
                                        relief_m=relief,
                                        floor_depth_m=shell_plan.depth_m, obstacles=obstacles)
        if props:
            poses, blocked = drop_blocked_cameras(poses, wall)
            layout_record["cameras_dropped"] += blocked

    exposure_evs = sample_exposure_evs(seed, len(poses), args.exposure_ev_range)
    # Partial white balance keeps a coloured wall coloured.
    _awb_rng = random.Random(int(seed) ^ 0x415742)
    awb_strengths = [round(_awb_rng.uniform(*args.awb_strength), 3) for _ in poses]
    configure_cycles(
        resolution=args.resolution,
        samples=args.samples,
        denoise=not args.no_denoise,
        dicing_rate=args.dicing_rate,
    )
    aov_names = setup_passes()

    # The record keeps the v1.17.0 schema plus prop_room; fixed settings are written as constants.
    record = {
        "schema": "syndefect3d/cwg_scene/1",
        "seed": seed,
        "spec": spec.to_dict(),
        "defects_placed": [p.to_dict() for p in placements],
        "defects_injected": injected,
        "props": [p.to_dict() for p in props],
        "prop_room": prop_room,
        "wall_relief_m": relief,
        "structures": structures,
        "junctions": junctions,
        "structure_settings": {"profile": args.structures, "views": 0, "kinds": None, "clean_fraction": 0.0},
        "room_shell": shell_plan.to_dict() if outdoor_plan is None else None,
        "shell_finish": shell_finish,
        "layout": layout_record,
        "environment": environment,
        "outdoor": outdoor_plan.to_dict() if outdoor_plan is not None else None,
        "weathering": weathering,
        "wall_material": {
            "source": "unified",
            "generator": generator_name,
            "surface_finish": surface_finish,
            "diverse": material_record,
            "clean_wall": clean_wall,
        },
        "lighting": lighting.to_dict(),
        "cameras": [p.to_dict() for p in poses],
        "camera_exposure_ev": exposure_evs,
        "light_profile": args.light_profile,
        "relight_per_view": args.relight_per_view,
        "camera_awb_strength": awb_strengths,
        "capture_settings": {
            "exposure_ev_range": args.exposure_ev_range,
            "closeup_fill": args.closeup_fill,
            "max_yaw_deg": args.closeup_max_yaw,
            "max_pitch_deg": args.closeup_max_pitch,
            "preserve_closeup_scale": True,
            "aim_jitter": args.closeup_aim_jitter,
        },
        "crack_field_settings": {
            "profile": args.crack_profile,
            "network_fraction": 0.0,
            "network_spacing_m": [.02, .06],
            "network_grime": args.crack_network_grime,
            "network_edge": "organic",
            "network_extent": args.crack_network_extent,
            "width_scale": args.crack_width_scale,
            "layered_fraction": args.crack_layered_fraction,
            "render": args.crack_render,
            "jitter_scale": 1.0,
        },
        "legacy_defect_names": False,
        "aov_names": list(aov_names),
        "resolution": [args.resolution, args.resolution],
        "samples": args.samples,
        "appearance_controls": {
            "crack_label_mode": "support",
            "crack_branch_mode": "local_curved",
            "crack_path_mode": "angular",
            "recipe": "unified",
            "layout_order": "fixtures-first",
            "crack_style": args.crack_style,
            "prop_placement": "loose",
            "crack_orientation": "vertical",
            "crack_junction_fraction": args.crack_junction_fraction,
            "crack_weathering_fraction": args.crack_weathering_fraction,
            "crack_axis_snap": 0.0,
            "paint_profile": "baseline",
            "wall_normal_mode": "explicit",
            "surface_detail": False, "camera_isp": args.camera_isp,
            "crack_depth_scale": args.crack_depth_scale, "crack_fit_path": True,
        },
    }

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(scene_dir / "scene.blend"))

    if not args.no_render:
        frames_dir.mkdir(parents=True, exist_ok=True)
        # The exporter pairs a frame with its scene manifest by directory, so
        # the per-scene instance manifest has to sit beside the passes too.
        if manifest_path.is_file():
            (frames_dir / "defect_instances.json").write_text(
                manifest_path.read_text()
            )
        relight = None
        if args.relight_per_view:
            record["view_lighting"] = []

            def relight(index, cam):
                if outdoor_plan is not None:  # outdoors there is one sun for the whole wall
                    record["view_lighting"].append(lighting.to_dict())
                    return
                # A fresh rig per view, aimed where this camera looks on the wall
                # (y = 0), from its own stream so nothing else is resampled.
                fwd = cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
                loc = cam.matrix_world.translation
                t = -loc.y / fwd.y if abs(fwd.y) > 1e-6 else 0.0
                aim = loc + fwd * max(t, 0.0)
                view_plan = sample_lighting_plan(
                    random.Random((seed << 8) ^ index ^ 0x52454C), spec.width_m, spec.height_m,
                    profile=args.light_profile, target=(aim.x, 0.0, aim.z))
                apply_lighting(view_plan, spec.height_m, capture_collection)
                record["view_lighting"].append(view_plan.to_dict())

        t_render = time.perf_counter()
        record["frames"] = render_cameras(
            [p.object_name for p in poses], frames_dir, aov_names=aov_names,
            exposure_evs=exposure_evs, awb_strengths=awb_strengths, before_each=relight,
        )
        if args.camera_isp:
            from syncrack3d.isp import process_frames

            record["camera_isp"] = process_frames(frames_dir, record["frames"], seed)

    t_end = time.perf_counter()
    record["timing_s"] = {"build": round((t_render or t_end) - t_start, 2),
                          "render": round(t_end - t_render, 2) if t_render else None,
                          "views": len(record.get("frames", []))}
    (scene_dir / "scene.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run = parser.add_argument_group("run")
    run.add_argument("--seeds", default="0-9", help="e.g. 0-49 or 1,4,9")
    run.add_argument("--output", required=True)
    run.add_argument("--resolution", type=int, default=640)
    run.add_argument("--samples", type=int, default=64)
    run.add_argument("--dicing-rate", type=float, default=.75, help="displacement dicing rate in pixels")
    run.add_argument("--no-denoise", action="store_true")
    run.add_argument("--no-render", action="store_true", help="build scenes only")
    run.add_argument("--save-blend", action="store_true")
    run.add_argument("--keep-going", action="store_true", help="log and continue when one seed fails")
    run.add_argument("--resume", action="store_true", help="skip seeds whose scene.json exists (it is written last)")

    data = parser.add_argument_group("data design and ablations")
    data.add_argument("--crack-style-mix", default=DEFAULT_MIX,
                      help="per-wall crack styles with weights (styles.py), e.g. legacy=1 for one style")
    data.add_argument("--material-categories", nargs="+", choices=list(MATERIAL_CATEGORIES), metavar="CATEGORY",
                      help="draw the wall material from these categories only (default: all 15): "
                           + ", ".join(MATERIAL_CATEGORIES))
    data.add_argument("--clean-wall-scale", type=float, default=0.0,
                      help="multiplier on the per-category crack-free wall rates (0: every wall cracked)")
    data.add_argument("--clean-wall-uniform", type=float, default=None,
                      help="one crack-free probability for every material instead of the per-category "
                           "table (same per-wall draw); 1 renders every wall crack-free")
    data.add_argument("--crack-render", choices=("geometry", "pbr", "albedo"), default="geometry",
                      help="real displacement, bump/normal map only (PBR decal), or colour only")
    data.add_argument("--crack-layered-fraction", type=float, default=0.7,
                      help="share of coated-host cracks with an edge: paint film lips/chips, or a broken "
                           "stucco coat. Brick, stone, concrete and tile stay a groove (0 disables)")
    data.add_argument("--crack-near-service-fraction", type=float, default=0.3,
                      help="share of crack tokens tried beside a cable/pipe/trunking run")
    data.add_argument("--no-fixtures", action="store_true",
                      help="no wall fixtures; indoor rooms then keep the sampled open shell")
    data.add_argument("--shell-finish", choices=("plain", "matched"), default="matched",
                      help="indoor context: plain flat colours, or side walls in the test wall's finish "
                           "(70%%, else both in one other wall category) and an Infinigen floor material")
    data.add_argument("--max-props", type=int, default=3,
                      help="props per indoor wall (0 disables)")
    data.add_argument("--light-profile", choices=LIGHT_PROFILES, default="default",
                      help="default: sampled indoor regimes; hard_mix adds sun, bare-bulb and raking-torch "
                           "sources; fixed: one constant soft key light indoors (outdoor walls keep their sun)")
    data.add_argument("--relight-per-view", action="store_true",
                      help="sample a new light rig per view, aimed where the camera looks (needs hard_mix)")
    data.add_argument("--exposure-ev-range", type=float, nargs=2, default=(.4, 1.3), metavar=("MIN", "MAX"),
                      help="sensor exposure compensation per view (-3..3 EV)")
    data.add_argument("--awb-strength", type=float, nargs=2, default=(0.0, .5), metavar=("MIN", "MAX"),
                      help="per-camera grey-world white-balance strength (0..1)")
    data.add_argument("--camera-isp", action="store_true", help="phone-like ISP on the rendered frames")
    data.add_argument("--closeups-per-defect", type=int, default=2)
    data.add_argument("--closeup-fill", type=float, nargs=2, default=(.6, 1.5), metavar=("MIN", "MAX"),
                      help="requested card-width fraction of the frame (not mask area)")
    data.add_argument("--closeup-max-yaw", type=float, default=38.0)
    data.add_argument("--closeup-max-pitch", type=float, default=26.0)
    data.add_argument("--closeup-aim-jitter", type=float, default=.3,
                      help="shift close-up aim by up to this many frame widths (0-0.5)")
    data.add_argument("--context-view-fraction", type=float, default=0.25)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    try:
        parse_style_mix(args.crack_style_mix)
    except ValueError as exc:
        parser.error(str(exc))
    if args.clean_wall_uniform is not None and not 0 <= args.clean_wall_uniform <= 1:
        parser.error("--clean-wall-uniform must be 0..1")
    if not 0 <= args.crack_layered_fraction <= 1 or not 0 <= args.crack_near_service_fraction <= 1:
        parser.error("--crack-layered-fraction and --crack-near-service-fraction must be 0..1")
    if args.relight_per_view and args.light_profile == "default":
        parser.error("--relight-per-view needs --light-profile hard_mix")
    if not 0 < args.closeup_fill[0] <= args.closeup_fill[1] <= 2:
        parser.error("--closeup-fill requires 0 < MIN <= MAX <= 2 (above 1 overfills the frame)")
    if not 0 <= args.closeup_aim_jitter <= 0.5:
        parser.error("--closeup-aim-jitter must be in [0, 0.5]")
    if not 0 <= args.closeup_max_yaw < 90 or not 0 <= args.closeup_max_pitch < 90:
        parser.error("camera angles must lie in [0, 90) degrees")
    if not -3 <= args.exposure_ev_range[0] <= args.exposure_ev_range[1] <= 3:
        parser.error("exposure range requires -3 <= MIN <= MAX <= 3")
    if not 0 <= args.awb_strength[0] <= args.awb_strength[1] <= 1:
        parser.error("white-balance strength requires 0 <= MIN <= MAX <= 1")

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s | %(message)s"
    )
    out_root = Path(args.output).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    ok = failed = skipped = total_defects = total_frames = 0
    for seed in seeds:
        if args.resume and (out_root / f"seed_{seed:05d}" / "scene.json").is_file():
            skipped += 1
            continue
        try:
            record = build_one(seed, wall_style(seed, args)[1], out_root)
        except Exception:
            failed += 1
            logger.error("seed %d failed:\n%s", seed, traceback.format_exc())
            # A seed that fails mid-render (e.g. GPU OOM on one close-up) leaves
            # the cameras it finished beside its manifest, which the exporter
            # reads as a complete scene. Move them out of its path.
            partial = out_root / "frames" / f"seed_{seed:05d}"
            if partial.exists():
                quarantine = out_root / "frames_failed" / partial.name
                if quarantine.exists():
                    shutil.rmtree(quarantine)
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                partial.rename(quarantine)
            if not args.keep_going:
                raise
            continue
        ok += 1
        total_defects += record["defects_injected"]
        total_frames += len(record.get("frames", []))
        logger.info(
            "seed %d: %d defect(s), %d prop(s), %d camera(s), regime %s",
            seed,
            record["defects_injected"],
            len(record["props"]),
            len(record["cameras"]),
            record["lighting"]["regime"],
        )

    print(
        f"\n{ok} scene(s) built, {failed} failed, {skipped} already built; "
        f"{total_defects} defects injected, {total_frames} frames -> {out_root}"
    )
    return 1 if failed and not args.keep_going else 0


if __name__ == "__main__":
    sys.exit(main())
