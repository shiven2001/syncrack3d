#!/usr/bin/env python3
"""SynCrack3D dataset driver: build, inject, light, pose, render.

One seed produces one wall, a sampled lighting regime, a few props, several
phone-like cameras, and one rendered frame per camera with crack AOVs beside
it. SynCrack3D places cracks only. Paint peeling and blistering stay in
``controlled_wall_generator`` and ``infinigen``.

    python syncrack3d/build_dataset.py --seeds 0-49 --output out/walls

Run it with the same Python that has bpy installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
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

from syncrack3d.cameras import sample_camera_poses  # noqa: E402
from syncrack3d.defects import (  # noqa: E402
    build_defects,
    inject_defects,
    sample_defect_plan,
)
from syncrack3d.framing import sample_exposure_evs  # noqa: E402
from syncrack3d.geometry import build_scene  # noqa: E402
from syncrack3d.materials import enable_true_displacement  # noqa: E402
from syncrack3d.infinigen_materials import (  # noqa: E402
    apply_infinigen_wall_material,
)
from syncrack3d.lighting import (  # noqa: E402
    LIGHT_PROFILES,
    apply_lighting,
    sample_lighting_plan,
)
from syncrack3d.structures import (  # noqa: E402
    PROFILES as STRUCTURE_PROFILES,
    KINDS as STRUCTURE_KINDS,
    build_structures,
    sample_structure_plan,
    structure_camera_targets,
    structure_clean_wall,
)
from syncrack3d.occlusion import sample_props  # noqa: E402
from syncrack3d.outdoor import (  # noqa: E402
    apply_sky,
    apply_weathering,
    build_outdoor,
    environment_for,
    outdoor_lighting,
    remove_indoor_props,
    sample_outdoor_plan,
)
from syncrack3d.room_shell import (  # noqa: E402
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
from syncrack3d.spec import SUBSTRATES, sample_scene_spec  # noqa: E402
from syncrack3d.paint_profiles import PAINT_PROFILES, adapt_paint_material  # noqa: E402
from syncrack3d.material_diversity import (  # noqa: E402
    CATEGORIES as MATERIAL_CATEGORIES,
    apply_diverse_wall_material,
    is_clean_wall,
    virtual_targets,
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
    scene_dir = out_root / f"seed_{seed:05d}"
    frames_dir = out_root / "frames" / f"seed_{seed:05d}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    spec = sample_scene_spec(seed, module="flat", substrate=args.substrate)
    if args.paint_profile != "baseline":
        if args.wall_materials != "controlled" or args.surface_detail:
            raise ValueError("--paint-profile requires controlled materials without --surface-detail")
        spec = replace(spec, material=adapt_paint_material(spec.material, seed, args.paint_profile))
    if args.surface_detail and args.wall_materials != "controlled":
        raise ValueError("--surface-detail requires --wall-materials controlled")
    if args.surface_detail and spec.material.finish == "paint":
        # Independent draw: do not change placement, lighting, or camera RNG.
        fine = random.Random(seed ^ 0x53555246).uniform(250, 650)
        spec = replace(spec, material=replace(
            spec.material, texture_scale=fine,
            color_variation=spec.material.color_variation * 2,
        ))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    rng = random.Random(seed)

    clear_scene()
    build = build_scene(spec)
    wall = build.hosts[0]
    if args.wall_normal_mode == "explicit":
        # CWG's camera-facing plane is at y=0, with geometric normal -Y.
        # Positive height must move toward cameras at negative Y; no sign flip.
        wall["syndefect_displacement_sign"] = 1
    # Shade the wall with Infinigen's own material, not the base generator's
    # stand-in. This is the surface apartment defects are injected into, so it
    # is the only choice under which the two datasets differ in one variable
    # rather than two - and injection reads the host shader, so a simpler one
    # silently takes different branches.
    material_record, clean_wall = None, False
    if args.wall_materials in ("infinigen", "diverse"):
        # Both material paths draw through numpy's global RNG, and its state here
        # depended on which seeds ran earlier in the same process (seed 106003
        # alone vs after 106002 gave a different Plaster colour, relief and
        # displacement). Cause: the material helpers import Infinigen's room
        # decorator lazily, and that first import draws from numpy, so only a
        # process's first wall saw the extra draws. Import first, then reseed,
        # so the wall is a function of its own seed only.
        import cwg_infinigen.core.constraints.example_solver.room.decorate  # noqa: F401
        from cwg_infinigen.core import surface  # noqa: F401
        from cwg_infinigen.core.util.math import int_hash

        np.random.seed(int_hash((seed, "cwg_wall_material")) % (2**32))
    if args.wall_materials == "infinigen":
        generator_name, surface_finish = apply_infinigen_wall_material(wall, rng)
    elif args.wall_materials == "diverse":
        # CWGv5: wide substrate/colour palette; non-paint walls mostly defect-free
        generator_name, surface_finish, material_record = apply_diverse_wall_material(
            wall, seed, args.material_category
        )
        clean_wall = is_clean_wall(seed, material_record["category"], args.clean_wall_scale)
    else:
        generator_name, surface_finish = "controlled", spec.material.finish
    clean_wall = clean_wall or structure_clean_wall(seed, args.structures, args.structure_clean_fraction)
    shell_plan = sample_shell_plan(rng)  # drawn for every wall: keeps the shared RNG stream
    environment = environment_for(seed, args.environment)
    outdoor_plan = sample_outdoor_plan(seed, spec.width_m) if environment == "outdoor" else None
    if outdoor_plan is None and args.structures != "off":
        # Fixture views reach wall boundaries; close the room so sockets and
        # door frames are not photographed against the sky outside the slab.
        shell_plan = replace(shell_plan, has_ceiling=True, returns=("left", "right"))
    if outdoor_plan is None:
        build_room_shell(spec, shell_plan)
    else:
        build_outdoor(spec, outdoor_plan, wall)
    configure_capture_scene(spec, build.root)
    capture_collection = bpy.data.collections[CAPTURE_COLLECTION]

    plan = sample_defect_plan(
        rng,
        total_range=(0, 0) if clean_wall else (args.min_defects, args.max_defects),
        substrate_finish=surface_finish,
        size_range_m=tuple(args.defect_size_range),
    )
    placements = build_defects(spec, wall, rng, plan, legacy_names=args.legacy_defect_names,
                               axis_snap=args.crack_axis_snap)
    props = sample_props(rng, spec, placements, count_range=(0, args.max_props))
    lighting = sample_lighting_plan(rng, spec.width_m, spec.height_m)
    if args.light_profile != "default" and not args.relight_per_view:
        # Independent stream: walls, defects and cameras stay as in the default run.
        lighting = sample_lighting_plan(random.Random(seed ^ 0x4C494748), spec.width_m, spec.height_m,
                                        profile=args.light_profile)
    apply_lighting(lighting, spec.height_m, capture_collection)
    weathering = {}
    if outdoor_plan is not None:
        # one sun (or overcast sky) per wall, visible sky; the indoor plan above
        # was still drawn so the shared RNG stream is unchanged
        lighting = outdoor_lighting(outdoor_plan, spec)
        apply_lighting(lighting, spec.height_m, capture_collection)
        apply_sky(outdoor_plan)
        weathering = apply_weathering(wall, outdoor_plan, spec.height_m)

    manifest_path = scene_dir / "defect_instances.json"
    import gin
    import cwg_infinigen.assets.defect_injection  # noqa: F401

    with gin.unlock_config():
        if args.crack_render != "geometry":
            gin.bind_parameter("_inject_crack.crack_render", args.crack_render)
        if args.crack_layered_fraction > 0:
            gin.bind_parameter("_inject_crack.layered_fraction", args.crack_layered_fraction)
        gin.bind_parameter("_inject_crack.crack_profile", args.crack_profile)
        gin.bind_parameter("_inject_crack.crack_network_fraction", args.crack_network_fraction)
        gin.bind_parameter("_inject_crack.crack_network_spacing", tuple(args.crack_network_spacing))
        gin.bind_parameter("_inject_crack.crack_network_grime", tuple(args.crack_network_grime))
        gin.bind_parameter("_inject_crack.crack_network_edge", args.crack_network_edge)
        gin.bind_parameter("_inject_crack.crack_network_extent", args.crack_network_extent)
        gin.bind_parameter("_inject_crack.crack_width_scale", args.crack_width_scale)
        gin.bind_parameter("_inject_crack.crack_jitter_scale", args.crack_jitter_scale)
        gin.bind_parameter("_inject_crack.crack_depth_scale", args.crack_depth_scale)
        gin.bind_parameter("_inject_crack.crack_fit_path", args.crack_fit_path)
        gin.bind_parameter("_inject_crack.crack_label_mode", args.crack_label_mode)
        gin.bind_parameter("_inject_crack.crack_branch_mode", args.crack_branch_mode)
        gin.bind_parameter("_inject_crack.crack_path_mode", args.crack_path_mode)
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

    structure_plans = sample_structure_plan(seed, spec, placements, args.structures, environment, args.structure_kinds)
    structures = build_structures(structure_plans)

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
        preserve_closeup_scale=args.preserve_closeup_scale,
        aim_jitter=args.closeup_aim_jitter,
    )
    if structures and args.structure_views:
        structure_rng = random.Random(seed ^ 0x53544341)
        targets = list(structures)
        structure_rng.shuffle(targets)
        poses.extend(sample_camera_poses(
            structure_rng, spec, structure_camera_targets(targets, args.structure_views),
            capture_collection, closeups_per_defect=1, frame_fill_range=(.45, .75),
            index_offset=len(poses), closeup_role="structure", include_context=False,
            max_yaw_deg=min(args.closeup_max_yaw, 25), max_pitch_deg=min(args.closeup_max_pitch, 20),
        ))
    if outdoor_plan is not None:
        remove_indoor_props()  # sampled for the RNG stream, but furniture is not outdoors

    exposure_evs = sample_exposure_evs(seed, len(poses), args.exposure_ev_range)
    # Partial white balance keeps a coloured wall coloured (opt-in; default full).
    awb_strengths = None
    if args.awb_strength is not None:
        _awb_rng = random.Random(int(seed) ^ 0x415742)
        awb_strengths = [round(_awb_rng.uniform(*args.awb_strength), 3) for _ in poses]
    configure_cycles(
        resolution=args.resolution,
        samples=args.samples,
        denoise=not args.no_denoise,
        dicing_rate=args.dicing_rate,
    )
    aov_names = setup_passes()

    record = {
        "schema": "syndefect3d/cwg_scene/1",
        "seed": seed,
        "spec": spec.to_dict(),
        "defects_placed": [p.to_dict() for p in placements],
        "defects_injected": injected,
        "props": [p.to_dict() for p in props],
        "structures": structures,
        "structure_settings": {"profile": args.structures, "views": args.structure_views, "kinds": args.structure_kinds,
                               "clean_fraction": args.structure_clean_fraction},
        "room_shell": shell_plan.to_dict() if outdoor_plan is None else None,
        "environment": environment,
        "outdoor": outdoor_plan.to_dict() if outdoor_plan is not None else None,
        "weathering": weathering,
        "wall_material": {
            "source": args.wall_materials,
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
            "preserve_closeup_scale": args.preserve_closeup_scale,
            "aim_jitter": args.closeup_aim_jitter,
        },
        "crack_field_settings": {
            "profile": args.crack_profile,
            "network_fraction": args.crack_network_fraction,
            "network_spacing_m": args.crack_network_spacing,
            "network_grime": args.crack_network_grime,
            "network_edge": args.crack_network_edge,
            "network_extent": args.crack_network_extent,
            "width_scale": args.crack_width_scale,
            "layered_fraction": args.crack_layered_fraction,
            "render": args.crack_render,
            "jitter_scale": args.crack_jitter_scale,
        },
        "legacy_defect_names": args.legacy_defect_names,
        "aov_names": list(aov_names),
        "resolution": [args.resolution, args.resolution],
        "samples": args.samples,
        "appearance_controls": {
            "crack_label_mode": args.crack_label_mode,
            "crack_branch_mode": args.crack_branch_mode,
            "crack_path_mode": args.crack_path_mode,
            "crack_axis_snap": args.crack_axis_snap,
            "paint_profile": args.paint_profile,
            "wall_normal_mode": args.wall_normal_mode,
            "surface_detail": args.surface_detail, "camera_isp": args.camera_isp,
            "crack_depth_scale": args.crack_depth_scale, "crack_fit_path": args.crack_fit_path,
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

        record["frames"] = render_cameras(
            [p.object_name for p in poses], frames_dir, aov_names=aov_names,
            exposure_evs=exposure_evs, awb_strengths=awb_strengths, before_each=relight,
        )
        if args.camera_isp:
            from syncrack3d.isp import process_frames

            record["camera_isp"] = process_frames(frames_dir, record["frames"], seed)

    (scene_dir / "scene.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-9", help="e.g. 0-49 or 1,4,9")
    parser.add_argument("--output", required=True)
    parser.add_argument("--substrate", default="random", choices=("random",) + SUBSTRATES)
    parser.add_argument(
        "--wall-materials",
        default="infinigen",
        choices=("infinigen", "controlled", "diverse"),
        help=(
            "Which shader the wall carries. 'infinigen' uses the same "
            "Plaster/Concrete/Brick/NonWoodTile palette the apartment walls do, "
            "so a controlled-wall dataset is comparable against the apartment "
            "one; 'controlled' keeps the base generator's deterministic stand-in."
        ),
    )
    parser.add_argument("--material-category", default=None, choices=sorted(MATERIAL_CATEGORIES),
                        help="with --wall-materials diverse: force one category (QA)")
    parser.add_argument("--clean-wall-scale", type=float, default=1.0,
                        help="with --wall-materials diverse: multiplier on per-category defect-free rates")
    parser.add_argument("--awb-strength", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="per-camera grey-world AWB strength (0..1); default full correction")
    parser.add_argument("--resolution", type=int, default=1280)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--wall-normal-mode", choices=("legacy", "explicit"), default="legacy")
    parser.add_argument("--surface-detail", action="store_true")
    parser.add_argument("--paint-profile", choices=PAINT_PROFILES, default="baseline")
    parser.add_argument("--camera-isp", action="store_true")
    parser.add_argument("--crack-depth-scale", type=float, default=1.0)
    parser.add_argument("--crack-fit-path", action="store_true")
    parser.add_argument("--crack-label-mode", choices=("profile", "support"), default="profile")
    parser.add_argument("--crack-branch-mode", choices=("original", "bounded_curved", "local_curved"), default="original")
    parser.add_argument("--crack-path-mode", choices=("original", "meander", "angular", "straight"), default="original")
    parser.add_argument("--crack-axis-snap", type=float, default=0.0,
                        help="share of cracks turned to run along a wall axis (+-2 deg), like joint cracks")
    parser.add_argument("--defect-size-range", type=float, nargs=2, default=(.10, .45))
    parser.add_argument("--environment", choices=("indoor", "outdoor", "mixed"), default="indoor",
                        help="indoor room, outdoor building facade (ground, sky, sun, weathering), or half each")
    parser.add_argument("--crack-render", choices=("geometry", "pbr", "albedo"), default="geometry",
                        help="ablation: real displacement, bump/normal map only (PBR decal), or colour only")
    parser.add_argument("--crack-layered-fraction", type=float, default=0.7,
                        help="share of coated-host cracks with an edge: paint film lips/chips, or a broken stucco coat. Brick, stone, concrete and tile stay a groove (0 disables)")
    parser.add_argument("--light-profile", choices=LIGHT_PROFILES, default="default",
                        help="hard_mix adds sun, bare-bulb and raking-torch sources (opt-in)")
    parser.add_argument("--relight-per-view", action="store_true",
                        help="sample a new light rig per view, aimed where the camera looks")
    parser.add_argument("--dicing-rate", type=float, default=2.0)
    parser.add_argument("--min-defects", type=int, default=3)
    parser.add_argument("--max-defects", type=int, default=6)
    parser.add_argument("--max-props", type=int, default=3)
    parser.add_argument("--structures", choices=STRUCTURE_PROFILES, default="off",
                        help="background cables/pipes/boxes/windows/joints; auto selects indoor/outdoor mix")
    parser.add_argument("--structure-kinds", nargs="+", choices=STRUCTURE_KINDS,
                        help="force a subset for QA; requires --structures enabled")
    parser.add_argument("--structure-views", type=int, default=2,
                        help="additional views aimed at structural details (0 disables)")
    parser.add_argument("--structure-clean-fraction", type=float, default=0.0,
                        help="fraction of structure-enabled scenes forced defect-free (0..1)")
    parser.add_argument("--crack-profile", choices=("legacy", "major", "mixed", "network"), default="legacy",
                        help="legacy mix, broad cracks, 50%% of each, or hairline cellular networks")
    parser.add_argument("--crack-network-fraction", type=float, default=0.0,
                        help="replace this fraction of legacy/major/mixed instances with crazing (0..1)")
    parser.add_argument("--crack-network-spacing", type=float, nargs=2, default=(.02, .06),
                        metavar=("MIN", "MAX"), help="network cell spacing range in metres")
    parser.add_argument("--crack-network-grime", type=float, nargs=2, default=(.15, .60), metavar=("MIN", "MAX"),
                        help="network channel staining, as a share of crack grime strength (0..1)")
    parser.add_argument("--crack-network-edge", choices=("square", "organic"), default="organic",
                        help="network patch outline: organic (v1.7.0) or square, which reproduces v1.4.0-v1.5.0")
    parser.add_argument("--crack-network-extent", type=float, default=1.4,
                        help="network field size relative to its card (.5-2); 1.4 keeps organic crack length near a square patch. Cards and cameras are unchanged. Use 1 with --crack-network-edge square to reproduce v1.4.0-v1.5.0")
    parser.add_argument("--crack-width-scale", type=float, default=1.3,
                        help="physical crack width multiplier (.25-4), also changes AOV")
    parser.add_argument("--crack-jitter-scale", type=float, default=1.0,
                        help="high-frequency path roughness multiplier (0-2)")
    parser.add_argument("--closeups-per-defect", type=int, default=1)
    parser.add_argument("--exposure-ev-range", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"), help="sensor exposure compensation per view (-3..3 EV); default unchanged")
    parser.add_argument("--closeup-fill", type=float, nargs=2, default=(0.25, 0.80),
                        metavar=("MIN", "MAX"), help="requested card-width fraction (not mask area)")
    parser.add_argument("--closeup-max-yaw", type=float, default=38.0)
    parser.add_argument("--closeup-max-pitch", type=float, default=26.0)
    parser.add_argument("--preserve-closeup-scale", action="store_true",
                        help="use longer lenses when minimum standoff would shrink the defect")
    parser.add_argument("--context-view-fraction", type=float, default=0.25)
    parser.add_argument("--closeup-aim-jitter", type=float, default=0.0,
                        help="shift close-up aim by up to this many frame widths (0-0.5)")
    parser.add_argument("--legacy-defect-names", action="store_true",
                        help="v1 card names: every scene reuses the same defect shapes")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--no-render", action="store_true", help="build scenes only")
    parser.add_argument("--save-blend", action="store_true")
    parser.add_argument("--keep-going", action="store_true",
                        help="log and continue when one seed fails")
    parser.add_argument("--resume", action="store_true",
                        help="skip seeds whose scene.json exists (it is written last)")
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None
    )
    if args.structure_kinds and args.structures == "off":
        parser.error("--structure-kinds requires --structures enabled")
    if not 0 <= args.structure_clean_fraction <= 1:
        parser.error("structure clean fraction must be 0..1")
    if args.structure_views < 0:
        parser.error("structure views must be nonnegative")
    if not 0 <= args.crack_layered_fraction <= 1:
        parser.error("--crack-layered-fraction must be 0-1")
    if args.relight_per_view and args.light_profile == "default":
        parser.error("--relight-per-view needs --light-profile hard_mix")
    if not .05 <= args.defect_size_range[0] <= args.defect_size_range[1] <= 1.0:
        parser.error("defect sizes require .05 <= MIN <= MAX <= 1 metre")
    if not 0 < args.closeup_fill[0] <= args.closeup_fill[1] <= 2:
        parser.error("--closeup-fill requires 0 < MIN <= MAX <= 2 (above 1 overfills the frame)")
    if not 0 <= args.closeup_aim_jitter <= 0.5:
        parser.error("--closeup-aim-jitter must be in [0, 0.5]")
    if not 0 <= args.closeup_max_yaw < 90 or not 0 <= args.closeup_max_pitch < 90:
        parser.error("camera angles must lie in [0, 90) degrees")
    if not 0 <= args.crack_network_fraction <= 1:
        parser.error("crack network fraction must be 0..1")
    if not 0 < args.crack_network_spacing[0] <= args.crack_network_spacing[1] < float("inf"):
        parser.error("crack network spacing must be positive and ordered")
    if not 0 <= args.crack_network_grime[0] <= args.crack_network_grime[1] <= 1:
        parser.error("crack network grime must be ordered within 0..1")
    if not .5 <= args.crack_network_extent <= 2:
        parser.error("crack network extent must be .5..2")
    if not .25 <= args.crack_width_scale <= 4 or not 0 <= args.crack_jitter_scale <= 2:
        parser.error("crack width scale must be .25..4; jitter scale must be 0..2")
    if not .05 <= args.crack_depth_scale <= 2:
        parser.error("crack depth scale must be .05..2")
    if args.crack_branch_mode != "original" and not args.crack_fit_path:
        parser.error("curved branches require --crack-fit-path")
    if args.exposure_ev_range is not None and not -3 <= args.exposure_ev_range[0] <= args.exposure_ev_range[1] <= 3:
        parser.error("exposure range requires -3 <= MIN <= MAX <= 3")

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
            record = build_one(seed, args, out_root)
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
