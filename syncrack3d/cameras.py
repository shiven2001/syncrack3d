"""Phone-like capture poses for the controlled wall.

The real test set is phone close-ups of a defect, usually handheld, usually not
square to the wall. The apartment pipeline approximated that with
``pose_defect_cameras``; here the wall is flat and its normal is known exactly,
so the pose can be sampled directly from the thing that matters - how far off
the surface normal the photographer was standing - rather than solved for.

Three deliberate choices:

* **Off-axis by default.** A head-on shot of a flat wall is the one framing a
  person almost never takes, and it is also the framing under which a plane has
  no perspective cue at all. Yaw and pitch are sampled to +-38 degrees.
* **Distance follows the defect, not the wall.** Framing is chosen so the defect
  fills a sampled fraction of the frame, so a hairline and a wide crack stay
  usable after the downscale to 640.
* **Some frames are context, not close-ups.** A model trained only on centred
  defects loses precision on clean wall, so a fraction of rigs pull back to show
  the surface with its skirting, props and lighting falloff.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import bpy
from mathutils import Euler, Vector

from syncrack3d.framing import solve_closeup

# A phone main camera. Matches camera.pose_defect_cameras.sensor_width_mm in the
# apartment configs so both datasets share an intrinsics family.
SENSOR_WIDTH_MM = 5.6
_LENS_EQUIV_MM = (16.0, 24.0, 26.0, 35.0, 52.0, 77.0)
_LENS_WEIGHTS = (0.08, 0.26, 0.24, 0.20, 0.15, 0.07)
_FULL_FRAME_DIAG_MM = 36.0


@dataclass(frozen=True)
class CameraPose:
    index: int
    role: str
    target_index: int | None
    focal_mm: float
    lens_equiv_mm: float
    distance_m: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    frame_fill: float
    location: tuple[float, float, float]
    object_name: str
    nominal_extent_fill: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _sample_focal(rng):
    equiv = rng.choices(_LENS_EQUIV_MM, weights=_LENS_WEIGHTS, k=1)[0]
    return SENSOR_WIDTH_MM * equiv / _FULL_FRAME_DIAG_MM, equiv


def _pose_camera(name, location, target, roll_deg, collection):
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.object
    cam.name = name
    cam.data.name = f"{name}_Data"
    cam.data.sensor_width = SENSOR_WIDTH_MM
    direction = Vector(target) - Vector(location)
    euler = direction.to_track_quat("-Z", "Y").to_euler("XYZ")
    if abs(roll_deg) > 1e-6:
        # Handheld tilt. Applied about the view axis so it rotates the frame
        # rather than re-aiming it.
        euler = (
            euler.to_matrix().to_4x4()
            @ Euler((0.0, 0.0, math.radians(roll_deg)), "XYZ").to_matrix().to_4x4()
        ).to_euler("XYZ")
    cam.rotation_euler = euler
    for old in tuple(cam.users_collection):
        old.objects.unlink(cam)
    collection.objects.link(cam)
    return cam


def sample_camera_poses(
    rng: random.Random,
    spec,
    placements,
    collection,
    *,
    closeups_per_defect: int = 1,
    context_view_fraction: float = 0.25,
    max_yaw_deg: float = 38.0,
    max_pitch_deg: float = 26.0,
    frame_fill_range: tuple[float, float] = (0.25, 0.80),
    distance_clamp_m: tuple[float, float] = (0.22, 3.2),
    preserve_closeup_scale: bool = False,
    aim_jitter: float = 0.0,
    index_offset: int = 0,
    closeup_role: str = "closeup",
    include_context: bool = True,
) -> list[CameraPose]:
    """One or more close-ups per crack, plus a share of pulled-back views."""
    poses: list[CameraPose] = []
    # fill > 1 frames the defect larger than the image, so it runs off the edges.
    if not 0 < frame_fill_range[0] <= frame_fill_range[1] <= 2:
        raise ValueError("frame_fill_range must satisfy 0 < min <= max <= 2")
    if not 0 <= aim_jitter <= 0.5:
        raise ValueError("aim_jitter must be in [0, 0.5] frame widths")
    index = index_offset

    for placement in placements:
        for _ in range(closeups_per_defect):
            focal_mm, equiv = _sample_focal(rng)
            fill = rng.uniform(*frame_fill_range)
            target = Vector((placement.u_m, 0.0, placement.v_m))
            extent = placement.size_m
            # Horizontal field of view at distance d covers
            # d * sensor_width / focal. Solve it for the distance that makes
            # the defect occupy `fill` of the frame width.
            distance = placement.size_m * focal_mm / (SENSOR_WIDTH_MM * fill)
            if preserve_closeup_scale:
                focal_mm, distance, _ = solve_closeup(
                    extent, fill, focal_mm, SENSOR_WIDTH_MM, distance_clamp_m,
                    [SENSOR_WIDTH_MM * lens / _FULL_FRAME_DIAG_MM for lens in _LENS_EQUIV_MM],
                )
                equiv = focal_mm * _FULL_FRAME_DIAG_MM / SENSOR_WIDTH_MM
            if aim_jitter > 0:
                # Handheld, off-centre framing: shift the aim point by up
                # to aim_jitter frame widths, so defects sit off-centre or
                # partly outside the image. No draws when disabled.
                frame_w = extent / fill
                target = Vector(
                    (
                        placement.u_m + rng.uniform(-aim_jitter, aim_jitter) * frame_w,
                        0.0,
                        placement.v_m + rng.uniform(-aim_jitter, aim_jitter) * frame_w,
                    )
                )
            distance = min(max(distance, distance_clamp_m[0]), distance_clamp_m[1])

            yaw = math.radians(rng.uniform(-max_yaw_deg, max_yaw_deg))
            pitch = math.radians(rng.uniform(-max_pitch_deg, max_pitch_deg))
            # -Y is the wall's outward normal; yaw swings around Z, pitch lifts.
            offset = Vector(
                (
                    math.sin(yaw) * math.cos(pitch),
                    -math.cos(yaw) * math.cos(pitch),
                    math.sin(pitch),
                )
            ) * distance
            if closeup_role == "structure":
                # Boundary trim is near floor/ceiling: keep the camera inside
                # the usable wall height rather than below the floor or roof.
                offset.x = max(-spec.width_m*.5+.08, min(spec.width_m*.5-.08, target.x+offset.x)) - target.x
                yaw = math.atan2(offset.x, -offset.y)
                offset.z = max(.08, min(spec.height_m - .08, target.z + offset.z)) - target.z
                distance = offset.length
                pitch = math.atan2(offset.z, math.hypot(offset.x, offset.y))
            location = target + offset
            roll = rng.uniform(-7.0, 7.0)
            name = f"CWG_Cam_{index:03d}_{closeup_role}"
            cam = _pose_camera(name, tuple(location), tuple(target), roll, collection)
            cam.data.lens = focal_mm
            cam["cwg_role"] = closeup_role
            if closeup_role == "closeup":
                cam["cwg_target_index"] = placement.index
                cam["syndefect_target"] = placement.object_name
            else:
                cam["cwg_structure_target"] = placement.object_name
            poses.append(
                CameraPose(
                    index=index,
                    role=closeup_role,
                    target_index=placement.index if closeup_role == "closeup" else None,
                    focal_mm=round(focal_mm, 4),
                    lens_equiv_mm=equiv,
                    distance_m=round(float(distance), 4),
                    yaw_deg=round(math.degrees(yaw), 2),
                    pitch_deg=round(math.degrees(pitch), 2),
                    roll_deg=round(roll, 2),
                    frame_fill=round(fill, 3),
                    location=tuple(round(float(v), 5) for v in location),
                    object_name=name,
                    nominal_extent_fill=round(extent * focal_mm / (SENSOR_WIDTH_MM * distance), 4),
                )
            )
            index += 1

    n_context = max(1, int(round(len(poses) * context_view_fraction))) if poses else 2
    for _ in range(n_context if include_context else 0):
        focal_mm, equiv = _sample_focal(rng)
        distance = rng.uniform(1.2, 3.4)
        yaw = math.radians(rng.uniform(-max_yaw_deg, max_yaw_deg))
        pitch = math.radians(rng.uniform(-18.0, 14.0))
        aim_u = rng.uniform(-spec.width_m * 0.3, spec.width_m * 0.3)
        aim_v = rng.uniform(spec.height_m * 0.25, spec.height_m * 0.75)
        target = Vector((aim_u, 0.0, aim_v))
        offset = Vector(
            (
                math.sin(yaw) * math.cos(pitch),
                -math.cos(yaw) * math.cos(pitch),
                math.sin(pitch),
            )
        ) * distance
        location = target + offset
        roll = rng.uniform(-5.0, 5.0)
        name = f"CWG_Cam_{index:03d}_context"
        cam = _pose_camera(name, tuple(location), tuple(target), roll, collection)
        cam.data.lens = focal_mm
        cam["cwg_role"] = "context"
        poses.append(
            CameraPose(
                index=index,
                role="context",
                target_index=None,
                focal_mm=round(focal_mm, 4),
                lens_equiv_mm=equiv,
                distance_m=round(float(distance), 4),
                yaw_deg=round(math.degrees(yaw), 2),
                pitch_deg=round(math.degrees(pitch), 2),
                roll_deg=round(roll, 2),
                frame_fill=0.0,
                location=tuple(round(float(v), 5) for v in location),
                object_name=name,
            )
        )
        index += 1

    if poses:
        bpy.context.scene.camera = bpy.data.objects[poses[0].object_name]
    return poses
