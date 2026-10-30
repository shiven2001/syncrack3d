# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma, Lahav Lipson: Stationary camera selection
# - Alexander Raistrick: Refactor into proposal/validate, camera animation
# - Lingjie Mei: get_camera_trajectory


import json
import logging
import typing
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path

import bpy
import gin
import imageio
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from numpy.random import uniform as U
from tqdm import tqdm

from cwg_infinigen.core.nodes import node_utils
from cwg_infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from cwg_infinigen.core.rendering.post_render import colorize_depth
from cwg_infinigen.core.tagging import tag_system
from cwg_infinigen.core.util import blender as butil
from cwg_infinigen.core.util import camera
from cwg_infinigen.core.util.blender import SelectObjects, delete
from cwg_infinigen.core.util.logging import Timer
from cwg_infinigen.core.util.organization import SelectionCriterions
from cwg_infinigen.core.util.random import random_general
from cwg_infinigen.terrain.core import Terrain
from cwg_infinigen.tools.suffixes import get_suffix

from . import animation_policy

logger = logging.getLogger(__name__)


@gin.configurable
def get_sensor_coords(cam, H=None, W=None, sparse=False):
    camd = cam.data
    f_in_m = camd.lens / 1000
    scene = bpy.context.scene
    if W is None:
        W = int(scene.render.resolution_x)
    if H is None:
        H = int(scene.render.resolution_y)
    resolution_x_in_px = W
    resolution_y_in_px = H

    scale = scene.render.resolution_percentage / 100
    sensor_width_in_m = camd.sensor_width / 1000
    sensor_height_in_m = camd.sensor_height / 1000
    assert abs(sensor_width_in_m / sensor_height_in_m - W / H) < 1e-4, (
        sensor_width_in_m,
        sensor_height_in_m,
        W,
        H,
    )

    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if camd.sensor_fit == "VERTICAL":
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = (
            resolution_x_in_px * scale / sensor_width_in_m / pixel_aspect_ratio
        )  # pixels per milimeter
        s_v = resolution_y_in_px * scale / sensor_height_in_m

    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_m
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_m

    u_0 = resolution_x_in_px * scale / 2  # cx (in pixels) Usually is just W/2
    v_0 = resolution_y_in_px * scale / 2  # cx (in pixels) Usually is just H/2
    xx, yy = np.meshgrid(np.arange(W).astype(float), np.arange(H).astype(float))
    coords_x = (xx - u_0) / s_u  # relative, in mm
    coords_y = (yy - v_0 + 1) / s_v  # relative, in mm

    coords_z = np.full(coords_x.shape, -f_in_m)
    relative_cam_coords = np.stack((coords_x, coords_y, coords_z), axis=-1)

    cam_coords_vectors = np.empty((H, W), dtype=Vector)
    pixel_locs = np.stack((np.meshgrid(np.arange(W), np.arange(H))), axis=-1).reshape(
        (W * H, 2)
    )  # np.array(list(product(range(H), range(W))))
    if sparse:
        ii = np.random.choice(H * W, size=1000)
        pixel_locs = pixel_locs[ii]

    for x, y in tqdm(pixel_locs, desc="Building Camera Vectors", disable=True):
        pixelVector = Vector(relative_cam_coords[y, x])
        cam_coords_vectors[y, x] = cam.matrix_world @ pixelVector

    return cam_coords_vectors, pixel_locs


@gin.configurable
def adjust_camera_sensor(cam, sensor_width=None, sensor_height=18.0):
    """Match camera sensor aspect to the current render resolution.

    Default is original Infinigen (18 mm height, width scaled to aspect).
    Phone-like configs pass ``sensor_width`` (e.g. 5.6 mm iPhone 11) and height
    is derived so the K-matrix aspect check stays valid.
    """
    scene = bpy.context.scene
    W = scene.render.resolution_x
    H = scene.render.resolution_y
    if sensor_width is not None:
        cam.data.sensor_fit = "HORIZONTAL"
        cam.data.sensor_width = float(sensor_width)
        cam.data.sensor_height = float(sensor_width) * H / W
        return
    cam.data.sensor_fit = "AUTO"
    cam.data.sensor_height = float(sensor_height)
    cam.data.sensor_width = float(sensor_height) * (W / H)


@gin.configurable
def configure_smartphone_optics(
    cam,
    enabled=False,
    focal_length=3.7,
    use_dof=True,
    aperture_fstop=1.8,
    focus_distance=2.0,
    clip_start=0.02,
    apply_focal_length=True,
    apply_focus_distance=True,
):
    """iPhone-like lens / deep DoF. No-op unless ``enabled`` (see realism_v2.gin)."""
    if not enabled or cam is None or cam.type != "CAMERA":
        return
    if apply_focal_length:
        cam.data.lens = float(focal_length)
    cam.data.clip_start = float(clip_start)
    cam.data.dof.use_dof = bool(use_dof)
    cam.data.dof.aperture_fstop = float(aperture_fstop)
    if apply_focus_distance:
        cam.data.dof.focus_distance = float(focus_distance)


def spawn_camera():
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.clip_end = 1e4
    adjust_camera_sensor(cam)
    configure_smartphone_optics(cam)
    return cam


def cam_name(cam_rig, subcam):
    return f"camera_{cam_rig}_{subcam}"


def get_id(camera: bpy.types.Object):
    _, rig, subcam = camera.name.split("_")
    return int(rig), int(subcam)


@gin.configurable
def spawn_camera_rigs(
    camera_rig_config,
    n_camera_rigs,
    start_index: int = 0,
) -> list[bpy.types.Object]:
    """Spawn camera rig empties ``camrig.{i}`` with child cameras ``camera_i_j``.

    Use ``start_index`` to append rigs after existing ones (e.g. defect-focus extras).
    """
    rigs_col = butil.get_collection("camera_rigs")
    cams_col = butil.get_collection("cameras")

    def spawn_rig(i):
        rig_parent = butil.spawn_empty(f"camrig.{i}")
        butil.put_in_collection(rig_parent, rigs_col)

        for j, config in enumerate(camera_rig_config):
            cam = spawn_camera()
            cam.name = cam_name(i, j)
            cam.parent = rig_parent
            cam.location = config["loc"]
            cam.rotation_euler = config["rot_euler"]

            butil.put_in_collection(cam, cams_col)

        return rig_parent

    return [spawn_rig(i) for i in range(start_index, start_index + n_camera_rigs)]


@gin.configurable
def add_defect_focus(enabled: bool = False, max_rigs: int = 24):
    """If True, spawn extra close-up camera rigs for defects after the main batch.

    Apartment scenes place 100+ defects. One stereo rig per defect, then
    duplicating those dense grids into the camera BVH, OOMs the coarse stage.
    ``max_rigs`` caps extras; sampling is stratified by kind. ``max_rigs < 0``
    means unlimited.
    """
    return enabled, max_rigs


def save_camera_rig_manifest(cam_rigs, path, room_rigs=()) -> dict:
    """Record what each rig is for, next to the scene that owns it.

    Rig role cannot be read off the index: room_view_balance appends its extra
    room rigs *after* the defect close-ups, so camrig.6 can be a room view while
    camrig.1 is a close-up. Anything downstream that guesses from the numbering
    silently skips cameras. The pose diagnostics ride along because they are the
    per-view provenance the manifest otherwise lacks - which lens, how far back,
    how oblique, and how the light was hitting the defect.
    """
    room_names = {r.name for r in room_rigs}
    records = []
    for rig in cam_rigs:
        suffix = rig.name.split(".")[-1]
        index = int(suffix) if suffix.isdigit() else 0
        target = rig.get("syndefect_target")
        record = {
            "rig": rig.name,
            "index": index,
            "role": "defect_closeup" if target else "room_trajectory",
        }
        if target:
            record.update(
                {
                    "target": target,
                    "pose_valid": bool(rig.get("syndefect_pose_valid", False)),
                    "pose_reason": rig.get("syndefect_pose_reason"),
                    "distance_m": rig.get("syndefect_view_distance_m"),
                    "view_angle_deg": rig.get("syndefect_view_angle_deg"),
                    "focal_mm": rig.get("syndefect_focal_mm"),
                    "lens_equiv_mm": rig.get("syndefect_lens_equiv_mm"),
                    "frame_fill": rig.get("syndefect_frame_fill"),
                    "light_incidence_deg": rig.get("syndefect_light_incidence_deg"),
                }
            )
        elif rig.name in room_names:
            record["role"] = "room_trajectory"
        records.append(record)

    payload = {
        "schema": "syndefect3d/camera_rigs/1",
        "counts": {
            "room_trajectory": sum(r["role"] == "room_trajectory" for r in records),
            "defect_closeup": sum(r["role"] == "defect_closeup" for r in records),
        },
        "rigs": sorted(records, key=lambda r: r["index"]),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "wrote %s: %d room / %d defect close-up",
        path.name,
        payload["counts"]["room_trajectory"],
        payload["counts"]["defect_closeup"],
    )
    return payload


@gin.configurable
def room_view_balance(
    n_defect_rigs: int,
    n_existing_room_rigs: int = 1,
    room_view_fraction: float = 0.30,
    enabled: bool = True,
) -> int:
    """How many extra room-trajectory rigs to add alongside the close-ups.

    Close-ups are the positives and they match how inspection photos are taken,
    but a detector trained only on centred defects never sees clean plaster,
    grout, seams, sockets or skirting and has no way to learn what is *not* a
    defect. Room views are that negative and context stream.

    The count is derived from the number of close-ups rather than fixed, so the
    mix does not drift as the defect count varies between scenes.

    Note this balances *rigs*, not frames: a room rig carries an animated
    trajectory and can contribute several frames, so the frame-level mix depends
    on how many frames each rig is rendered for.
    """
    if not enabled or room_view_fraction <= 0.0 or n_defect_rigs <= 0:
        return 0
    if room_view_fraction >= 1.0:
        raise ValueError("room_view_fraction must be < 1")
    want = n_defect_rigs * room_view_fraction / (1.0 - room_view_fraction)
    return max(0, int(round(want)) - int(n_existing_room_rigs))


def sample_focus_targets(objs, seed: int, max_n: int):
    """Pick up to ``max_n`` objects, round-robin across ``syndefect_kind`` / factory."""
    objs = [o for o in objs if o is not None]
    if max_n is None or max_n < 0 or len(objs) <= max_n:
        return objs
    rng = np.random.default_rng(int(seed) + 17)

    def bucket_key(obj):
        kind = obj.get("syndefect_kind") if hasattr(obj, "get") else None
        if kind:
            return str(kind)
        return (obj.name or "defect").split("(")[0]

    buckets: dict[str, list] = {}
    for obj in objs:
        buckets.setdefault(bucket_key(obj), []).append(obj)
    for group in buckets.values():
        rng.shuffle(group)
    keys = list(buckets)
    rng.shuffle(keys)
    picked = []
    while keys and len(picked) < max_n:
        still = []
        for key in keys:
            if buckets[key] and len(picked) < max_n:
                picked.append(buckets[key].pop())
            if buckets[key]:
                still.append(key)
        keys = still
    return picked


def get_camera_rigs() -> list[bpy.types.Object]:
    if "camera_rigs" not in bpy.data.collections:
        raise ValueError("No camera rigs found")

    result = list(bpy.data.collections["camera_rigs"].objects)

    for i, rig in enumerate(result):
        for j, child in enumerate(rig.children):
            expected = cam_name(i, j)
            if child.name != expected:
                raise ValueError(f"child {i=} {j}  was {child.name=}, {expected=}")

    return result


@node_utils.to_nodegroup(
    "nodegroup_camera_info", singleton=True, type="GeometryNodeTree"
)
def nodegroup_active_cam_info(nw: NodeWrangler):
    info = nw.new_node(Nodes.ObjectInfo, [bpy.context.scene.camera])
    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={k: info.outputs[k] for k in info.outputs.keys()},
    )


def set_active_camera(camera: bpy.types.Object):
    bpy.context.scene.camera = camera

    ng = (
        nodegroup_active_cam_info()
    )  # does not create a new node group, retrieves singleton
    ng.nodes["Object Info"].inputs["Object"].default_value = camera

    return bpy.context.scene.camera


def terrain_camera_query(
    cam: bpy.types.Object,
    scene_bvh: BVHTree,
    terrain_tags_queries,
    vertexwise_min_dist,
    min_dist=0,
):
    dists = []
    sensor_coords, pix_it = get_sensor_coords(cam, sparse=True)
    terrain_tags_queries_counts = {q: 0 for q in terrain_tags_queries}

    for x, y in pix_it:
        direction = (sensor_coords[y, x] - cam.matrix_world.translation).normalized()
        _, _, index, dist = scene_bvh.ray_cast(cam.matrix_world.translation, direction)
        if dist is None:
            continue
        dists.append(dist)
        if dist < min_dist or (
            vertexwise_min_dist is not None and dist < vertexwise_min_dist[index]
        ):
            logger.debug(f"Found {dist=} < {min_dist=}")
            dists = None  # means dist < min
            break
        for q in terrain_tags_queries:
            terrain_tags_queries_counts[q] += terrain_tags_queries[q][index]

    n_pix = pix_it.shape[0]

    return dists, terrain_tags_queries_counts, n_pix


@dataclass
class CameraProposal:
    loc: np.array
    rot: np.array
    focal_length: float

    def apply(self, cam_rig):
        cam_rig.location = self.loc
        cam_rig.rotation_euler = self.rot

        if self.focal_length is not None:
            for cam in cam_rig.children:
                cam.data.lens = self.focal_length


@gin.configurable
def camera_pose_proposal(
    scene_bvh,
    location_sample: typing.Callable | tuple,
    center_coordinate=None,
    radius=None,
    bbox=None,
    altitude=("uniform", 1.5, 2.5),
    roll=0,
    yaw=("uniform", -180, 180),
    pitch=90,
    focal_length=50,
    override_loc=None,
):
    if isinstance(location_sample, tuple):
        location_sample = Vector(location_sample)

        def location_sample():
            return location_sample

    if override_loc is not None:
        loc = Vector(random_general(override_loc))
    elif center_coordinate:
        while True:
            # Define the radius of the circle
            random_angle = np.random.uniform(2 * np.math.pi)
            xoff = np.random.uniform(-radius / 10, radius / 10)
            yoff = np.random.uniform(-radius / 10, radius / 10)
            zoff = random_general(altitude)
            loc = Vector([0, 0, 0])
            loc[0] = center_coordinate[0] + radius * np.math.cos(random_angle) + xoff
            loc[1] = center_coordinate[1] + radius * np.math.sin(random_angle) + yoff
            loc[2] = center_coordinate[2] + zoff
            if bbox is not None:
                out_of_bbox = False
                for i in range(3):
                    if loc[i] < bbox[0][i] or loc[i] > bbox[1][i]:
                        out_of_bbox = True
                        break
                if out_of_bbox:
                    continue
            hit, *_ = scene_bvh.ray_cast(
                loc,
                Vector(center_coordinate) - loc,
                (Vector(center_coordinate) - loc).length,
            )
            if hit is None:
                break
    elif altitude is None:
        loc = location_sample()
    else:
        loc = location_sample()
        curr_alt = animation_policy.get_altitude(loc, scene_bvh)
        if curr_alt is None:
            logger.debug(f"camera_pose_proposal got {curr_alt=} for {loc=}")
            # butil.spawn_empty("fail")
            return None
        desired_alt = random_general(altitude)
        loc[2] = loc[2] + desired_alt - curr_alt

    if center_coordinate:
        direction = loc - Vector(center_coordinate)
        direction = Vector(direction)
        rotation_matrix = direction.to_track_quat("Z", "Y").to_matrix()
        rotation_euler = rotation_matrix.to_euler("XYZ")
        roll, pitch, yaw = rotation_euler
        noise_range = np.deg2rad(5.0)  # 5 degrees of noise in radians
        # Add random noise to roll, pitch, and yaw
        roll += np.random.uniform(-noise_range, noise_range)
        pitch += np.random.uniform(-noise_range, noise_range)
        yaw += np.random.uniform(-noise_range, noise_range)
        rot = np.array([roll, pitch, yaw])
    else:
        rot = np.deg2rad(
            [random_general(pitch), random_general(roll), random_general(yaw)]
        )
    focal_length = random_general(focal_length)
    return CameraProposal(loc, rot, focal_length)


@gin.configurable
def keep_cam_pose_proposal(
    cam: bpy.types.Object,
    terrain: Terrain,
    scene_bvh: BVHTree,
    placeholders_kd,
    camera_selection_answers,
    vertexwise_min_dist,
    camera_selection_ratio,
    min_placeholder_dist=0,
    min_terrain_distance=0,
    terrain_coverage_range=(0.5, 1),
):
    if terrain is not None:  # TODO refactor
        terrain_sdf = terrain.compute_camera_space_sdf(
            np.array(cam.matrix_world.translation).reshape((1, 3))
        )

    if not cam.type == "CAMERA":
        raise ValueError(f"{cam.name=} had {cam.type=}")

    bpy.context.view_layer.update()

    # Reject cameras too close to any placeholder vertex
    v, i, dist_to_placeholder = placeholders_kd.find(cam.matrix_world.translation)
    if dist_to_placeholder is not None and dist_to_placeholder < min_placeholder_dist:
        logger.debug(f"keep_cam_pose_proposal rejects {dist_to_placeholder=}, {v, i}")
        return None

    dists, camera_selection_answers_counts, n_pix = terrain_camera_query(
        cam,
        scene_bvh,
        camera_selection_answers,
        vertexwise_min_dist,
        min_dist=min_terrain_distance,
    )

    if dists is None:
        logger.debug("keep_cam_pose_proposal rejects terrain dists")
        return None

    coverage = len(dists) / n_pix
    if terrain_coverage_range is not None and (
        coverage < terrain_coverage_range[0]
        or coverage > terrain_coverage_range[1]
        or coverage == 0
    ):
        logger.debug(
            f"keep_cam_pose_proposal rejects {coverage=} for {terrain_coverage_range=}"
        )
        return None

    if terrain is not None and terrain_sdf <= 0:
        logger.debug(
            f"keep_cam_pose_proposal rejects {terrain_sdf=} for {cam.matrix_world.translation=}"
        )
        return None

    if rparams := camera_selection_ratio:
        for q in rparams:
            if type(q) is tuple and q[0] == SelectionCriterions.CloseUp:
                closeup = len([d for d in dists if d < q[1]]) / n_pix
                if closeup < rparams[q][0] or closeup > rparams[q][1]:
                    logger.debug(f"keep_cam_pose_proposal rejects {closeup=} for {q=}")
                    return None
            else:
                minv, maxv = rparams[q][0], rparams[q][1]
                if q in camera_selection_answers_counts:
                    ratio = camera_selection_answers_counts[q] / n_pix
                    if ratio < minv or ratio > maxv:
                        logger.debug(
                            f"keep_cam_pose_proposal rejects {ratio=} for {q=}"
                        )
                        return None

    return np.std(dists) + 1.5 * np.min(dists)


@gin.configurable
class AnimPolicyGoToProposals:
    def __init__(
        self, speed=("uniform", 1.5, 2.5), min_dist=4, max_dist=10, retries=30
    ):
        self.speed = speed
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.retries = retries

    def __call__(self, camera_rig, frame_curr, retry_pct, bvh):
        margin = Vector((self.max_dist, self.max_dist, self.max_dist))
        bbox = (camera_rig.location - margin, camera_rig.location + margin)

        for _ in range(self.retries):
            res = camera_pose_proposal(
                scene_bvh=bvh,
                location_sample=lambda: np.random.uniform(*bbox),
            )
            if res is None:
                continue
            dist = np.linalg.norm(np.array(res.loc) - np.array(camera_rig.location))
            if dist < self.min_dist:
                continue
            break
        else:
            raise animation_policy.PolicyError(
                f"{__name__} found no keyframe after {self.retries=}"
            )

        time = dist / random_general(self.speed)
        return Vector(res.loc), Vector(res.rot), time, "BEZIER"


@gin.configurable
def compute_base_views(
    camera_rig: bpy.types.Object,
    n_views: int,
    terrain,
    scene_bvh: BVHTree,
    location_sample: typing.Callable,
    center_coordinate=None,
    radius=None,
    bbox=None,
    placeholders_kd=None,
    min_candidates_ratio=20,
    max_tries=30000,
    visualize=False,
    **kwargs,
):
    potential_views = []
    n_min_candidates = int(min_candidates_ratio * n_views)

    with tqdm(total=n_min_candidates, desc="Searching for camera viewpoints") as pbar:
        for it in range(1, max_tries):
            if center_coordinate:
                props = camera_pose_proposal(
                    scene_bvh=scene_bvh,
                    location_sample=location_sample,
                    center_coordinate=center_coordinate,
                    radius=random_general(radius),
                    bbox=bbox,
                )
            else:
                props = camera_pose_proposal(
                    scene_bvh=scene_bvh, location_sample=location_sample
                )

            if props is None:
                logger.debug(
                    f"{camera_pose_proposal.__name__} returned {props=} for {it=}"
                )
                continue

            props.apply(camera_rig)

            all_scores = []
            for cam in camera_rig.children:
                score = keep_cam_pose_proposal(
                    cam,
                    terrain,
                    scene_bvh,
                    placeholders_kd,
                    **kwargs,
                )
                all_scores.append(score)

            if any(score is None for score in all_scores):
                criterion = None
            else:
                criterion = np.mean(all_scores)

            if visualize:
                criterion_str = f"{criterion:.2f}" if criterion is not None else "None"
                marker = butil.spawn_empty(f"attempt_{it}_{criterion_str}")
                marker.location = camera_rig.location
                marker.rotation_euler = camera_rig.rotation_euler

            if criterion is None:
                logger.debug(f"{it=} {criterion=}")
                continue

            # Compute focus distance
            destination = cam.matrix_world @ Vector((0.0, 0.0, -1.0))
            forward_dir = (destination - cam.location).normalized()
            *_, straight_ahead_dist = scene_bvh.ray_cast(cam.location, forward_dir)

            potential_views.append((criterion, deepcopy(props), straight_ahead_dist))
            pbar.update(1)

            if len(potential_views) >= n_min_candidates:
                break

    if len(potential_views) < n_views:
        if visualize:
            butil.save_blend("compute_base_views-failed.blend")
        raise ValueError(f"Could not find {n_views} camera views")

    views = sorted(potential_views, reverse=True)

    return views[:n_views]


def _defect_target_frame(obj):
    """Evaluated world matrix, geometry centroid and face normal for a defect.

    Defect assets are put in the `unique_assets` collection, which the view
    layer never evaluates, so their cached matrix_world stays at the identity
    however many depsgraph updates you run - every target then reports itself at
    the world origin and the close-up rigs get aimed at nothing. matrix_basis is
    recomputed from loc/rot/scale on access and is the whole answer for these,
    which are unparented.
    """
    matrix = obj.matrix_basis if obj.parent is None else obj.matrix_world
    if obj.type != "MESH" or obj.data is None or not obj.data.vertices:
        return matrix, Vector(matrix.translation), None
    centre = Vector((0.0, 0.0, 0.0))
    for vertex in obj.data.vertices:
        centre += matrix @ vertex.co
    centre /= len(obj.data.vertices)
    normal = Vector((0.0, 0.0, 0.0))
    normal_matrix = matrix.to_3x3().inverted().transposed()
    for polygon in obj.data.polygons:
        normal += normal_matrix @ polygon.normal
    if normal.length > 1e-8:
        normal.normalize()
    else:
        normal = None
    return matrix, centre, normal


# Phone lenses as 35 mm-equivalents with sampling weights. Real inspection
# photos come off a phone's main camera, sometimes its 2x, rarely the ultrawide.
# Equivalents rather than physical mm so a sensor change cannot silently turn a
# close-up into an 8 degree telephoto - which is exactly what a bare
# "focal_length = 40" against a 5.6 mm sensor did.
PHONE_LENS_EQUIV_MM = ((26.0, 0.65), (52.0, 0.25), (13.0, 0.10))
FULL_FRAME_WIDTH_MM = 36.0


def _sample_lens_mm(
    rng,
    sensor_width_mm,
    lens_table=PHONE_LENS_EQUIV_MM,
    extent=None,
    fill=None,
    distance_range=None,
):
    """Pick a lens that can actually frame this defect.

    Drawing the lens and the stand-off independently is what put an ultrawide on
    a five-centimetre blister cluster: framing it would need the camera closer
    than anyone can hold a phone, the distance clamps at the minimum, and the
    defect lands a few pixels across. A person choosing between the lenses on
    their phone does not do that - they reach for the 2x.

    So candidates whose required stand-off falls outside the handheld range are
    dropped before the draw, and the weights are renormalised over what is left.
    If nothing fits, the longest lens is used, which is the one that gets closest.
    """
    equivs = np.array([e for e, _ in lens_table], dtype=float)
    weights = np.array([w for _, w in lens_table], dtype=float)

    if extent is not None and fill is not None and distance_range is not None:
        focals = equivs * float(sensor_width_mm) / FULL_FRAME_WIDTH_MM
        needed = (float(extent) / float(fill)) * focals / float(sensor_width_mm)
        usable = (needed >= distance_range[0]) & (needed <= distance_range[1])
        if usable.any():
            equivs, weights = equivs[usable], weights[usable]
        else:
            equivs, weights = equivs[-1:], np.array([1.0])
            if len(lens_table) > 1:
                longest = int(np.argmax([e for e, _ in lens_table]))
                equivs = np.array([lens_table[longest][0]], dtype=float)

    equiv = float(rng.choice(equivs, p=weights / weights.sum()))
    return equiv * float(sensor_width_mm) / FULL_FRAME_WIDTH_MM, equiv


def _dominant_light_dir(point: Vector):
    """Direction the strongest light arrives from, and its irradiance weight.

    Crude on purpose: energy over distance squared, summed as vectors. It is
    used to describe the view, not to predict a pixel value.
    """
    total = Vector((0.0, 0.0, 0.0))
    best = 0.0
    for obj in bpy.context.scene.objects:
        if obj.type != "LIGHT" or obj.hide_render:
            continue
        light = obj.data
        energy = float(getattr(light, "energy", 0.0))
        if energy <= 0.0:
            continue
        if light.type == "SUN":
            direction = (obj.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
            weight = energy
        else:
            offset = obj.matrix_world.translation - point
            dist_sq = max(offset.length_squared, 1e-4)
            direction = offset.normalized()
            weight = energy / dist_sq
        total += direction * weight
        best = max(best, weight)
    if total.length < 1e-9:
        return None, 0.0
    return total.normalized(), best


def _outward_normal(point: Vector, normal: Vector, bvh, probe: float = 0.6):
    """Return whichever of +/-normal faces open space.

    Card winding is not reliable - several factories build planes whose local
    axis points into the host, and a defect with no recorded placement normal
    falls back to that axis. Backing a camera off along an inward normal puts it
    inside the wall, and the clearance ray then hits the host immediately and
    collapses the stand-off to its minimum. Probing both sides is cheap and
    removes the whole class of failure.
    """
    clearance = {}
    for sign in (1.0, -1.0):
        direction = normal * sign
        hit, _n, _i, dist = bvh.ray_cast(point + direction * 1e-3, direction, probe)
        clearance[sign] = probe if hit is None or dist is None else float(dist)
    if clearance[-1.0] > clearance[1.0] + 1e-3:
        return -normal, clearance[-1.0]
    return normal, clearance[1.0]


def _in_open_space(loc: Vector, bvh, clearance: float = 0.5, need: int = 2) -> bool:
    """Is the camera standing in a room, or buried in the building fabric?

    A line-of-sight test alone does not catch this. A camera inside a wall cavity
    or a stud bay can have a perfectly clear ray to a defect on the inner face of
    that same cavity - nothing is between them - and the render then comes back
    black. Probing the six axis directions separates the two: a camera in a room
    has open space in at least a couple of them, one buried in fabric has none.
    """
    open_dirs = 0
    for direction in (
        Vector((1, 0, 0)), Vector((-1, 0, 0)),
        Vector((0, 1, 0)), Vector((0, -1, 0)),
        Vector((0, 0, 1)), Vector((0, 0, -1)),
    ):
        hit, _n, _i, dist = bvh.ray_cast(loc + direction * 1e-3, direction, clearance)
        if hit is None or dist is None or float(dist) >= clearance:
            open_dirs += 1
            if open_dirs >= need:
                return True
    return False


def _project_px(point: Vector, cam_loc: Vector, rot_euler, focal_mm, sensor_mm, res):
    """Pinhole projection into the frame the sampled lens will actually render."""
    from mathutils import Euler

    basis = Euler(rot_euler, "XYZ").to_matrix()
    rel = basis.transposed() @ (point - cam_loc)
    depth = -rel.z
    if depth <= 1e-6:
        return None
    fx = float(focal_mm) / float(sensor_mm) * res[0]
    return (
        fx * (rel.x / depth) + res[0] / 2.0,
        -fx * (rel.y / depth) + res[1] / 2.0,
        depth,
    )


@gin.configurable
def pose_defect_cameras(
    cam_rigs: list[bpy.types.Object],
    defect_objs: list[bpy.types.Object],
    scene_bvh: BVHTree,
    distance: float | None = None,
    focal_length: float | None = None,
    sensor_width_mm: float = 5.6,
    frame_fill: tuple = (0.35, 0.95),
    distance_range: tuple = (0.3, 3.0),
    camera_height: tuple = (1.05, 1.70),
    max_azimuth_deg: float = 50.0,
    roll_deg: float = 2.5,
    aim_jitter_frac: float = 0.18,
    prefer_side_light: bool = True,
    min_defect_px: float = 24.0,
    max_frame_fill: float = 1.25,
    height_jitter: float = 0.0,
    horizontal_jitter: float = 0.0,
    angle_noise_deg: float = 0.0,
    max_retries: int = 50,
    one_to_one: bool = False,
    seed: int | None = None,
):
    """Place each rig where somebody photographing that defect would stand.

    The pose follows from the defect rather than from a fixed stand-off. A lens
    is drawn from the phone table, the distance is whatever frames the defect at
    a sampled fraction of the frame, and the camera sits at handheld height and
    tilts to the target instead of floating to the defect's own height. That
    couples the parameters the way a real photographer does: you step back for a
    metre-wide blister and crouch close to a hairline crack, and the vertical
    angle is a consequence of your height, not an independent random draw.

    Backwards compatible: an explicit ``distance`` or ``focal_length`` still
    wins, which keeps old configs reproducing their old poses.

    What this deliberately does not do is make a defect visible. Whether light
    rakes across relief is fixed by the light and the surface; a camera can only
    avoid shooting straight down the light axis and record the geometry it got.
    ``syndefect_light_incidence_deg`` is written to every rig for that reason -
    near 0 means the light is head-on and a dome-shaped defect will be a ghost
    whatever the camera does.
    """
    if not defect_objs:
        logger.warning("pose_defect_cameras: no defect objects found, skipping")
        return

    # Seeded from the scene, not from the clock. Lens, stand-off, height and
    # azimuth are all sampled here now, so an unseeded generator would hand the
    # same scene seed a different set of photographs on every coarse run and
    # quietly break the claim that a seed reproduces a dataset.
    rng = np.random.default_rng(seed)
    scene = bpy.context.scene
    res = (scene.render.resolution_x, scene.render.resolution_y)

    if one_to_one:
        targets_ordered = sorted(defect_objs, key=lambda o: o.name)
        n_pair = min(len(cam_rigs), len(targets_ordered))
        if len(cam_rigs) != len(targets_ordered):
            logger.warning(
                "pose_defect_cameras: one_to_one rig count %s != defect count %s, "
                "pairing first %s",
                len(cam_rigs),
                len(targets_ordered),
                n_pair,
            )
    else:
        shuffled = list(defect_objs)
        rng.shuffle(shuffled)
        targets_ordered = None

    for idx, cam_rig in enumerate(cam_rigs):
        if one_to_one:
            if idx >= n_pair:
                break
            target = targets_ordered[idx]
        else:
            target = shuffled[idx % len(shuffled)]
        target_mat, target_loc, _geometry_normal = _defect_target_frame(target)

        surface = target.get("syndefect_surface", "wall")
        # A defect may record the direction it actually faces. The local-axis
        # convention below only holds for planes built the wall/floor way, and
        # a decal seated on an arbitrary face has no reason to obey it.
        # Face winding differs between crack/peel/blister factories (some card
        # normals deliberately point into the host).  Camera stand-off follows
        # the explicit placement normal or the factory's surface convention,
        # never raw polygon winding.
        world_normal = None
        explicit = target.get("syndefect_normal")
        if explicit is not None and len(explicit) == 3:
            cand = Vector(tuple(float(c) for c in explicit))
            if cand.length > 1e-6:
                world_normal = cand.normalized()
        if world_normal is None:
            local_normal = Vector((1, 0, 0))
            if surface == "floor":
                local_normal = Vector((0, 0, 1))
            elif surface == "ceiling":
                local_normal = Vector((0, 0, -1))
            world_normal = (target_mat.to_3x3() @ local_normal).normalized()
        if world_normal.length < 1e-6:
            world_normal = Vector((0.0, 0.0, 1.0 if surface == "floor" else -1.0))
        if explicit is None:
            # Only the fallback axis needs checking; an explicitly placed normal
            # is already the direction the defect was seated facing.
            world_normal, _clear = _outward_normal(target_loc, world_normal, scene_bvh)

        # --- lens -------------------------------------------------------- #
        # --- how far back that lens has to be ---------------------------- #
        dims = sorted(float(d) for d in target.dimensions)
        extent = max(dims[-1], 1e-3)  # longest in-plane edge of the card
        fill = float(rng.uniform(*frame_fill))

        if focal_length is not None:
            focal_mm = float(focal_length)
            lens_equiv = focal_mm * FULL_FRAME_WIDTH_MM / float(sensor_width_mm)
        else:
            # Lens after the framing target, so it can be chosen to suit it.
            focal_mm, lens_equiv = _sample_lens_mm(
                rng,
                sensor_width_mm,
                extent=extent,
                fill=fill,
                distance_range=distance_range,
            )
        if distance is not None:
            want_dist = float(distance)
        else:
            want_dist = (extent / fill) * focal_mm / float(sensor_width_mm)
        want_dist = float(np.clip(want_dist, distance_range[0], distance_range[1]))

        # You cannot back further away than the room lets you. Hits within a
        # few cm are the host surface itself or a coincident face, not a wall
        # behind the photographer, so they must not drive the stand-off.
        backstop, _bl, _bn, back_dist = scene_bvh.ray_cast(
            target_loc + world_normal * 1e-3, world_normal
        )
        if backstop is not None and back_dist is not None and float(back_dist) > 0.1:
            want_dist = min(want_dist, max(float(back_dist) * 0.85, distance_range[0]))

        # --- where a person holding a phone would stand ------------------ #
        up = Vector((0.0, 0.0, 1.0))
        horiz_normal = Vector((world_normal.x, world_normal.y, 0.0))
        free_azimuth = horiz_normal.length < 1e-3  # floor / ceiling
        if free_azimuth:
            horiz_normal = Vector((1.0, 0.0, 0.0))
        else:
            horiz_normal.normalize()

        light_dir, _light_weight = _dominant_light_dir(target_loc)
        incidence_deg = None
        if light_dir is not None:
            incidence_deg = float(
                np.degrees(np.arccos(np.clip(abs(light_dir.dot(world_normal)), 0, 1)))
            )

        placed = False
        reason = "no_clear_pose_after_retries"
        cam_loc = target_loc + world_normal * want_dist
        rot_euler = (target_loc - cam_loc).to_track_quat("-Z", "Y").to_euler("XYZ")
        for attempt in range(max_retries):
            height = float(rng.uniform(*camera_height))
            dz = height - target_loc.z
            dz = float(np.clip(dz, -0.8 * want_dist, 0.8 * want_dist))
            radius = float(np.sqrt(max(want_dist**2 - dz**2, 1e-6)))

            if free_azimuth:
                azimuth = float(rng.uniform(-np.pi, np.pi))
            else:
                spread = np.radians(max_azimuth_deg)
                # Side-on beats shooting down the light axis: a defect lit from
                # behind the camera is flat-lit by construction.
                if prefer_side_light and light_dir is not None and attempt == 0:
                    light_h = Vector((light_dir.x, light_dir.y, 0.0))
                    if light_h.length > 1e-3:
                        light_h.normalize()
                        offset = np.arctan2(
                            light_h.cross(horiz_normal).z, light_h.dot(horiz_normal)
                        )
                        azimuth = float(np.clip(offset, -spread, spread))
                    else:
                        azimuth = float(rng.uniform(-spread, spread))
                else:
                    # Inspectors mostly face the wall; the tail is the exception.
                    azimuth = float(rng.triangular(-spread, 0.0, spread))

            offset_h = Vector(
                (
                    horiz_normal.x * np.cos(azimuth) - horiz_normal.y * np.sin(azimuth),
                    horiz_normal.x * np.sin(azimuth) + horiz_normal.y * np.cos(azimuth),
                    0.0,
                )
            ) * radius
            cand_loc = target_loc + offset_h + up * dz
            if horizontal_jitter or height_jitter:
                cand_loc += Vector(
                    (
                        rng.uniform(-horizontal_jitter, horizontal_jitter),
                        rng.uniform(-horizontal_jitter, horizontal_jitter),
                        rng.uniform(-height_jitter, height_jitter),
                    )
                )

            direction = target_loc - cand_loc
            if direction.length < 1e-4:
                continue
            if not _in_open_space(cand_loc, scene_bvh):
                reason = "camera_inside_geometry"
                continue
            hit, _hloc, _hn, hit_dist = scene_bvh.ray_cast(
                cand_loc, direction.normalized()
            )
            if hit is not None and hit_dist is not None and hit_dist < direction.length * 0.9:
                reason = "occluded"
                continue

            # Aim off-centre by a fraction of the frame: a handheld photo does
            # not put its subject on the principal point.
            aim = target_loc + Vector(
                (
                    rng.uniform(-1, 1) * extent * aim_jitter_frac,
                    rng.uniform(-1, 1) * extent * aim_jitter_frac,
                    rng.uniform(-1, 1) * extent * aim_jitter_frac,
                )
            )
            cand_euler = (aim - cand_loc).to_track_quat("-Z", "Y").to_euler("XYZ")
            if roll_deg:
                cand_euler.rotate_axis("Z", np.radians(rng.uniform(-roll_deg, roll_deg)))
            if angle_noise_deg > 0:
                noise = np.radians(angle_noise_deg)
                cand_euler.x += rng.uniform(-noise, noise)
                cand_euler.y += rng.uniform(-noise, noise)
                cand_euler.z += rng.uniform(-noise, noise)

            proj = _project_px(
                target_loc, cand_loc, cand_euler, focal_mm, sensor_width_mm, res
            )
            if proj is None:
                reason = "behind_camera"
                continue
            u, v, depth = proj
            if not (0 <= u < res[0] and 0 <= v < res[1]):
                reason = "target_out_of_frame"
                continue
            px = extent * (focal_mm / float(sensor_width_mm)) * res[0] / max(depth, 1e-6)
            if px < min_defect_px:
                reason = "defect_too_small_in_frame"
                continue
            if px > max_frame_fill * res[0]:
                # The frame is entirely inside the defect: no edge, no context,
                # and a ground-truth box clipped to the image border. Same
                # failure as the 40 mm lens, just less extreme.
                reason = "defect_overflows_frame"
                continue

            cam_loc, rot_euler = cand_loc, cand_euler
            placed = True
            reason = "clear_line_of_sight"
            break

        if not placed:
            # No pose can see this defect - it is enclosed, or every approach is
            # blocked. Do not leave the camera parked inside geometry: a frame
            # rendered from inside a wall is not a negative example, it is
            # garbage that would still enter the training pack. Retreat to the
            # most open direction so the rig at least yields a real view of the
            # room, and let require_qa_view drop the instance.
            best_dir, best_clear = None, 0.0
            for cand_dir in (
                Vector((1, 0, 0)), Vector((-1, 0, 0)),
                Vector((0, 1, 0)), Vector((0, -1, 0)),
                world_normal,
            ):
                hit, _n, _i, dist = scene_bvh.ray_cast(
                    target_loc + cand_dir * 1e-3, cand_dir, distance_range[1]
                )
                clear = distance_range[1] if hit is None or dist is None else float(dist)
                if clear > best_clear:
                    best_dir, best_clear = cand_dir, clear
            if best_dir is not None and best_clear > distance_range[0]:
                cam_loc = target_loc + best_dir * min(
                    best_clear * 0.8, distance_range[1]
                )
                rot_euler = (target_loc - cam_loc).to_track_quat("-Z", "Y").to_euler("XYZ")
            logger.warning(
                "pose_defect_cameras: could not find clear view for rig %s -> %s (%s), "
                "retreating to most open direction (%.2fm clear)",
                cam_rig.name,
                target.name,
                reason,
                best_clear,
            )

        dist_to_target = (target_loc - cam_loc).length
        view_angle = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        (cam_loc - target_loc).normalized().dot(world_normal), -1, 1
                    )
                )
            )
        )

        cam_rig.location = cam_loc
        cam_rig.rotation_euler = rot_euler
        cam_rig["syndefect_target"] = target.name
        cam_rig["syndefect_target_point"] = tuple(float(v) for v in target_loc)
        cam_rig["syndefect_pose_valid"] = bool(placed)
        cam_rig["syndefect_pose_reason"] = reason
        cam_rig["syndefect_view_distance_m"] = float(dist_to_target)
        cam_rig["syndefect_view_angle_deg"] = view_angle
        cam_rig["syndefect_focal_mm"] = float(focal_mm)
        cam_rig["syndefect_lens_equiv_mm"] = float(lens_equiv)
        cam_rig["syndefect_frame_fill"] = float(
            extent * (focal_mm / float(sensor_width_mm)) * res[0]
            / max(dist_to_target, 1e-6)
            / res[0]
        )
        if incidence_deg is not None:
            # 90 deg is grazing and reveals relief; near 0 is head-on and will
            # flatten a dome whatever the camera does.
            cam_rig["syndefect_light_incidence_deg"] = incidence_deg

        for cam in cam_rig.children:
            if cam.type != "CAMERA":
                continue
            cam.data.lens = focal_mm
            cam.data.sensor_width = float(sensor_width_mm)
            cam.data.dof.focus_distance = dist_to_target

        logger.info(
            "Defect camera %s -> %s at %.2fm, %.0fmm-equiv, %.0f deg off-normal, "
            "fill %.0f%%, light incidence %s",
            cam_rig.name,
            target.name,
            dist_to_target,
            lens_equiv,
            view_angle,
            100.0 * cam_rig["syndefect_frame_fill"],
            f"{incidence_deg:.0f} deg" if incidence_deg is not None else "unknown",
        )


def _meshes_for_bvh(objs):
    meshes = []
    for obj in objs:
        if obj is None:
            continue
        if getattr(obj, "type", None) != "MESH":
            continue
        data = getattr(obj, "data", None)
        if data is None or not getattr(data, "vertices", None):
            continue
        if len(data.vertices) == 0:
            continue
        meshes.append(obj)
    return meshes


_BOX_TRIS = (
    (0, 1, 2),
    (0, 2, 3),
    (4, 7, 6),
    (4, 6, 5),
    (0, 4, 5),
    (0, 5, 1),
    (3, 2, 6),
    (3, 6, 7),
    (0, 3, 7),
    (0, 7, 4),
    (1, 5, 6),
    (1, 6, 2),
)


def _bbox_tris_world(obj):
    verts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return verts, _BOX_TRIS


def _loop_tris_world(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    mw = obj.matrix_world
    verts = [mw @ v.co for v in mesh.vertices]
    faces = [tuple(tri.vertices) for tri in mesh.loop_triangles]
    return verts, faces


@gin.configurable
def build_bvh_and_attrs(objs, tags_queries, max_full_verts: int = 120000):
    """Raycast BVH without duplicating the scene in Blender.

    The old path duplicated every mesh, BEAUTY-triangulated, and joined. That
    OOMs furnished apartments (~10M verts). Indoor posing only needs walls and
    floors; face-tag answers are unused when tags_queries is empty.
    """
    objs = _meshes_for_bvh(objs)
    if not objs:
        raise ValueError("build_bvh_and_attrs: no mesh objects to join")

    all_verts = []
    all_faces = []
    n_full = n_box = 0
    for obj in objs:
        n_verts = len(obj.data.vertices)
        if n_verts > max_full_verts:
            verts, faces = _bbox_tris_world(obj)
            n_box += 1
        else:
            verts, faces = _loop_tris_world(obj)
            if not faces:
                verts, faces = _bbox_tris_world(obj)
                n_box += 1
            else:
                n_full += 1
        off = len(all_verts)
        all_verts.extend(verts)
        all_faces.extend(tuple(i + off for i in face) for face in faces)

    logger.info(
        "Building camera BVH from %s meshes (%s full, %s bbox, %s verts)",
        len(objs),
        n_full,
        n_box,
        len(all_verts),
    )
    bvh = BVHTree.FromPolygons(all_verts, all_faces, all_triangles=True)

    camera_selection_answers = {}
    for q0 in tags_queries:
        if type(q0) is not tuple:
            q = (q0,)
        else:
            q = q0
        if q[0] in [SelectionCriterions.CloseUp]:
            continue
        camera_selection_answers[q0] = np.zeros(0, dtype=bool)
    return bvh, camera_selection_answers


def camera_selection_preprocessing(
    terrain,
    scene_objs,
    tags_ratio: dict = None,
    ranges_ratio: dict = None,
    anim_criterion_keys: dict = None,
):
    if tags_ratio is None:
        tags_ratio = {}
    if ranges_ratio is None:
        ranges_ratio = {}
    if anim_criterion_keys is None:
        anim_criterion_keys = {}

    # preprocessing code adapted from mazeyu's original gin-oriented solution
    tags_ratio = {
        k: (*v, anim_criterion_keys.get(k, False)) for k, v in tags_ratio.items()
    }
    ranges_ratio = {
        v[:-2]: (v[-2], v[-1], anim_criterion_keys.get(k, False))
        for k, v in ranges_ratio.items()
    }

    all_selection_ratios = {**tags_ratio, **ranges_ratio}

    with Timer("Building placeholders KDTree"):
        placeholders = list(
            chain.from_iterable(
                c.all_objects
                for c in bpy.data.collections
                if c.name.startswith("placeholders:")
            )
        )
        placeholders = [p for p in placeholders if p.type == "MESH"]
        logger.info(f"Building placeholder kd for {len(placeholders)} objects")
        placeholders_kd = butil.joined_kd(placeholders, include_origins=True)

    if terrain is None:
        with Timer("Building scene BVH"):
            scene_bvh, camera_selection_answers = build_bvh_and_attrs(
                scene_objs, all_selection_ratios.keys()
            )
        vertexwise_min_dist = None
    else:
        scene_bvh, camera_selection_answers, vertexwise_min_dist = (
            terrain.build_terrain_bvh_and_attrs(all_selection_ratios.keys())
        )

    return dict(
        terrain=terrain,
        scene_bvh=scene_bvh,
        camera_selection_ratio=all_selection_ratios,
        camera_selection_answers=camera_selection_answers,
        vertexwise_min_dist=vertexwise_min_dist,
        placeholders_kd=placeholders_kd,
    )


@node_utils.to_nodegroup("geo_distrib", singleton=True, type="GeometryNodeTree")
def geo_distrib_random_points(nw: NodeWrangler):
    input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )
    distribute = nw.new_node(
        Nodes.DistributePointsOnFaces,
        input_kwargs={"Mesh": input.outputs["Geometry"], "Density": 500},
    )
    verts = nw.new_node(Nodes.PointsToVertices, [distribute])
    nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": verts})


def sample_random_locs(surface: bpy.types.Object, eps=0.01):
    # HACK implementation - uses blender geonodes' uniform surface sample, im fairly sure theres a numpy impl somewhere in the repo
    surface = butil.copy(surface)
    butil.apply_transform(surface, loc=True, rot=True, scale=True)
    butil.modify_mesh(
        surface, "NODES", node_group=geo_distrib_random_points(), apply=True
    )
    locs = np.array([v.co for v in surface.data.vertices])
    locs[:, -1] += eps
    butil.delete(surface)
    return locs


@gin.configurable
def configure_cameras(
    cam_rigs,
    scene_preprocessed: dict,
    init_bounding_box: tuple[np.array, np.array] = None,
    init_surfaces: list[bpy.types.Object] = None,
    terrain_mesh=None,
    nonroom_objs=None,
    mvs_setting=False,
    mvs_radius=("uniform", 12, 18),
    n_views: int = None,
    **kwargs,
):
    bpy.context.view_layer.update()

    if init_bounding_box is not None:

        def location_sample():
            return np.random.uniform(*init_bounding_box)
    elif init_surfaces is not None:
        random_locs = sample_random_locs(init_surfaces)

        def location_sample():
            loc = Vector(random_locs[np.random.randint(len(random_locs)), :])
            loc.z += 1e-3
            return loc
    else:
        raise ValueError("Either init_bounding_box or init_surfaces must be provided")

    if mvs_setting:
        if terrain_mesh:
            vertices = np.array([np.array(v.co) for v in terrain_mesh.data.vertices])
            sdfs = scene_preprocessed["terrain"].compute_camera_space_sdf(vertices)
            vertices = vertices[sdfs >= -1e-5]
            center_coordinate = list(
                vertices[np.random.choice(list(range(len(vertices))))]
            )
        elif nonroom_objs:

            def contain_keywords(name, keywords):
                for keyword in keywords:
                    if name == keyword or name.startswith(f"{keyword}."):
                        return True
                return False

            inside_objs = [
                x
                for x in nonroom_objs
                if not contain_keywords(x.name, ["window", "door", "entrance"])
            ]
            assert inside_objs != []
            obj = np.random.choice(inside_objs)
            vertices = [v.co for v in obj.data.vertices]
            center_coordinate = vertices[np.random.choice(list(range(len(vertices))))]
            center_coordinate = obj.matrix_world @ center_coordinate
            center_coordinate = list(np.array(center_coordinate))
        else:
            raise ValueError(
                f"Got {mvs_setting=} yet {terrain_mesh=} {nonroom_objs=}, we expected at least one in order to choose a center coordinate"
            )
    else:
        center_coordinate = None

    n = len(cam_rigs) if n_views is None else n_views
    views = None
    for cam_rig in cam_rigs:
        views = compute_base_views(
            cam_rig,
            n_views=n,
            location_sample=location_sample,
            center_coordinate=center_coordinate,
            radius=mvs_radius,
            bbox=init_bounding_box,
            **scene_preprocessed,
            **kwargs,
        )

        score, props, focus_dist = views[0]
        cam_rig.location = props.loc
        cam_rig.rotation_euler = props.rot

        for cam in cam_rig.children:
            cam.data.lens = props.focal_length

        if focus_dist is not None:
            for cam in cam_rig.children:
                if not cam.type == "CAMERA":
                    continue
                cam.data.dof.focus_distance = focus_dist

    return views


def anim_valid_camrig_pose_func(
    cam_rig: bpy.types.Object,
    scene_preprocessed: dict,
    animation_answers: dict,
    animation_ratio: dict,
    **kwargs,
):
    assert len(cam_rig.children) > 0

    scores = []

    for cam in cam_rig.children:
        score = keep_cam_pose_proposal(
            cam,
            placeholders_kd=scene_preprocessed["placeholders_kd"],
            scene_bvh=scene_preprocessed["scene_bvh"],
            terrain=scene_preprocessed["terrain"],
            vertexwise_min_dist=scene_preprocessed["vertexwise_min_dist"],
            camera_selection_answers=animation_answers,
            camera_selection_ratio=animation_ratio,
            **kwargs,
        )

        frame = bpy.context.scene.frame_current
        logger.debug(f"Checking {cam.name=} {frame=} got {score=}")

        if score is None:
            return None

        scores.append(score)

    return np.min(scores)


@gin.configurable
def animate_cameras(
    cam_rigs,
    bounding_box,
    scene_preprocessed,
    pois=None,
    follow_poi_chance=0.0,
    policy_registry=None,
    validate_pose_func=None,
    **kwargs,
):
    animation_ratio = {}
    animation_answers = {}
    for k in scene_preprocessed["camera_selection_ratio"]:
        if scene_preprocessed["camera_selection_ratio"][k][2]:
            animation_ratio[k] = scene_preprocessed["camera_selection_ratio"][k]
            animation_answers[k] = scene_preprocessed["camera_selection_answers"][k]

    if validate_pose_func is None:
        anim_valid_pose_func = partial(
            anim_valid_camrig_pose_func,
            scene_preprocessed=scene_preprocessed,
            camera_selection_answers=animation_answers,
            camera_selection_ratio=animation_ratio,
            **kwargs,
        )
    else:
        anim_valid_pose_func = anim_valid_camrig_pose_func

    for cam_rig in cam_rigs:
        if policy_registry is None:
            if U() < follow_poi_chance and pois is not None and len(pois):
                policy = animation_policy.AnimPolicyFollowObject(
                    target_obj=cam_rig, pois=pois, bvh=scene_preprocessed["scene_bvh"]
                )
            else:
                with gin.config_scope("cam"):
                    policy = animation_policy.AnimPolicyRandomWalkLookaround()
        else:
            policy = policy_registry()

        logger.info(f"Animating {cam_rig=} using {policy=}")

        animation_policy.animate_trajectory(
            cam_rig,
            scene_preprocessed["scene_bvh"],
            policy_func=policy,
            validate_pose_func=anim_valid_pose_func,
            verbose=True,
            fatal=True,
            bounding_box=bounding_box,
        )


@gin.configurable
def save_camera_parameters(
    camera_obj: bpy.types.Object, output_folder: Path, frame: int, use_dof=None
):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    if frame is not None:
        bpy.context.scene.frame_set(frame)

    camrig_id, subcam_id = get_id(camera_obj)

    if use_dof is not None:
        camera_obj.data.dof.use_dof = use_dof

    # Saving camera parameters
    K = camera.get_calibration_matrix_K_from_blender(camera_obj.data)
    suffix = get_suffix(
        dict(cam_rig=camrig_id, resample=0, frame=frame, subcam=subcam_id)
    )
    output_file = output_folder / f"camview{suffix}.npz"

    height_width = np.array(
        (
            bpy.context.scene.render.resolution_y,
            bpy.context.scene.render.resolution_x,
        )
    )
    T = np.asarray(camera_obj.matrix_world, dtype=np.float64) @ np.diag(
        (1.0, -1.0, -1.0, 1.0)
    )  # Y down Z forward (aka opencv)
    np.savez(output_file, K=np.asarray(K, dtype=np.float64), T=T, HW=height_width)


if __name__ == "__main__":
    """
    This interactive section generates a depth map by raycasting through each pixel. 
    It is very useful for debugging camera.py.
    """

    cams = [
        cam
        for cam in bpy.context.scene.objects
        if "cam" in cam.name.lower() and cam.type != "CAMERA"
    ]

    animate_cameras(cams)
    path = bpy.data.filepath
    bpy.ops.wm.save_mainfile(filepath=path)

    cam = bpy.context.scene.camera

    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080

    adjust_camera_sensor(cam)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    bvhtree = BVHTree.FromObject(bpy.context.active_object, depsgraph)

    target_obj = bpy.context.active_object
    to_obj_coords = target_obj.matrix_world.inverted()
    sensor_coords, pix_it = get_sensor_coords(cam, sparse=False)

    H, W = sensor_coords.shape
    depth_output = np.zeros((H, W), dtype=np.float64)

    for x, y in tqdm(pix_it):
        destination = sensor_coords[y, x]
        direction = (destination - cam.location).normalized()
        location, normal, index, dist = bvhtree.ray_cast(cam.location, direction)
        if dist is not None:
            dist_diff = (destination - cam.location).length
            assert dist > (location - destination).length, (
                dist,
                (location - destination).length,
            )
            assert dist > dist_diff
            depth_output[H - y - 1, x] = dist - dist_diff

    color_depth = colorize_depth(depth_output)
    imageio.imwrite("color_depth.png", color_depth)


@gin.configurable
def autofocus_cameras(
    enabled=True,
    circle_of_confusion_div=1500.0,
    min_distance_m=0.12,
    fallback_distance_m=3.0,
    respect_defect_targets=True,
):
    """Focus every camera on whatever it is actually pointing at.

    Two different things were leaving cameras mis-focused.

    Close-up rigs already compute their own stand-off and write it onto the
    camera, but `configure_smartphone_optics` ran afterwards at render time and
    overwrote it with a flat 2 m. That is fixed by `apply_focus_distance=False`,
    which then exposed the second problem: with nothing setting focus for the
    room-trajectory rigs, they kept whatever `compute_base_views` returned -
    measured at 0.29 m on a room camera, i.e. focused 29 cm in front of the lens
    while the room it is photographing is 1-5 m away. At f/1.8 that is the whole
    frame out of focus.

    Here a ray is cast down each camera's axis and the hit distance becomes the
    focus distance, which is what a phone's centre-weighted autofocus does.

    The result is then clamped to the hyperfocal distance,
    ``H = f^2 / (N * c) + f``. Focusing beyond H buys nothing - everything from
    H/2 to infinity is already acceptably sharp - and on a 3.7 mm phone lens H
    is only about 2.1 m, so a room view focused at the far wall and one focused
    at hyperfocal look the same while the latter also keeps the near corner
    sharp.

    Defect rigs are left alone by default: their target is the subject, and it
    is deliberately *not* the thing at the centre of the frame once aim jitter
    and roll have been applied.
    """
    if not enabled:
        return

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    focused = kept = 0
    for cam in bpy.data.objects:
        if cam.type != "CAMERA":
            continue
        rig = cam.parent
        if (
            respect_defect_targets
            and rig is not None
            and rig.get("syndefect_view_distance_m") is not None
        ):
            kept += 1
            continue

        data = cam.data
        coc = float(data.sensor_width) / float(circle_of_confusion_div)
        lens = float(data.lens)
        hyperfocal = lens * lens / (data.dof.aperture_fstop * coc) + lens
        hyperfocal /= 1000.0  # lens and coc are mm, scene is metres

        origin = cam.matrix_world.translation
        forward = (cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
        hit, location, _, _, _, _ = scene.ray_cast(depsgraph, origin, forward)
        distance = float((location - origin).length) if hit else None
        if distance is None or distance < min_distance_m:
            # No hit, or a hit so close the camera is inside geometry - one
            # room rig ray-cast at 0.12 m. Focusing there would be a worse
            # answer than not focusing at all, so fall back to hyperfocal,
            # which is sharp from H/2 to infinity.
            distance = hyperfocal
        data.dof.focus_distance = float(min(distance, hyperfocal))
        focused += 1

    logger.info(
        "autofocus: focused %d camera(s), left %d defect rig(s) on their target",
        focused,
        kept,
    )
