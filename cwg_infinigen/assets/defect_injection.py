# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Move solver-placed defects off their cards and into the host surface.

The constraint solver still places a defect plane exactly as before - nothing in
home.py, semantics.py or the gin configs changes. What changes is what happens
at finalize time: the plane is used only as a placement token. Its position and
orientation are read, the defect's shape is rebuilt as a field inside a copy of
the host wall's own material, Cycles displaces it into real geometry, and the
plane itself is deleted.

Doing it this way is what makes these three defects behave the way real ones do
under changing light. A card cannot recess, so a groove modelled on one reads as
a raised welt and its own silhouette betrays it at grazing angles; measured on a
plaster wall, a card seam shows at ~19-30 8-bit code values even in the best
configuration. Injected, the crack is a real notch in the wall, the wall's own
plaster grain runs unbroken across it, and its contrast swings from below 8-bit
quantisation head-on to clearly visible at 12 degrees.

Labels come from AOVs rather than the material index, because once a defect is a
field inside the wall's shader there is no separate material for the index pass
to find. See core/rendering/defect_aov.py.
"""

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import bpy
import gin
import numpy as np
from mathutils import Vector

from cwg_infinigen.assets.defect_fields import (
    build_crack_field,
    crack_layer_mode,
    sample_crack_layer,
    sample_crack_morphology,
)
from cwg_infinigen.core.rendering.defect_aov import (
    DEFECT_AOV_CLASSES,
    DEFECT_INSTANCE_AOV,
    DEFECT_INSTANCE_COVERAGE_AOV,
    DEFECT_INSTANCE_SQUARED_AOV,
)

logger = logging.getLogger(__name__)

# kind -> the material-name prefix its label should be filed under, so the AOV
# names stay identical to the plane-based path and the exporter needs no change.
KIND_PREFIX = {
    "crack": "CrackMaterial",
}


@dataclass
class HostInjection:
    """Accumulated defect fields for one host object."""

    host: bpy.types.Object
    material: bpy.types.Material
    height_terms: list = field(default_factory=list)
    class_fields: dict = field(default_factory=dict)
    # (coverage socket, instance id) per injected defect, for the instance AOV.
    instance_fields: list = field(default_factory=list)
    count: int = 0
    # Host geometric normals point out of the room (typical wall shells).
    # Injector heights assume positive is proud toward the occupant.
    flip_displacement: bool = False


_REGISTRY: dict[str, HostInjection] = {}

# One record per placed defect, injected or left as a card. Written out by
# write_defect_manifest so downstream work can identify instances rather than
# only classes.
_INSTANCES: list[dict] = []
_RUN_METADATA: dict = {}

# How much defect shader one host material can carry before labels start to go.
#
# Each material compiles to a fixed-size SVM stack. Past a certain graph size
# Cycles prints `out of SVM stack space, shader ... too big` and truncates the
# shader instead of failing the render. Truncation is progressive, not a switch:
# a few warnings cost nothing observable, and past some point the AOV label
# outputs stop being evaluated while the wall still shades at full brightness -
# a render that succeeds, looks correct, and labels nothing.
#
# The warning count is NOT the signal. Measured end to end on a real scene, by
# rendering a defect close-up and comparing the instances geometry says are
# visible against the instances actually present in the instance AOV:
#
#   cap  injected  SVM warnings  labels
#     6        18             3  all expected instances present
#     8        24             7  none - every label lost
#    99        31             9  none - every label lost
#
# So 3 warnings coexist with perfect labels and 7 warnings means total loss.
# Anything that changes shader size - a different realism overlay, heavier host
# materials, more AOVs - moves this, and the only trustworthy check is
# expected-versus-labelled, not the log. An earlier version of this comment
# claimed any truncation drops labels; that was inferred from a warning and an
# unrelated empty mask, and is wrong.
MAX_DEFECTS_PER_HOST = 6

# Stop injecting into a host material once it reaches this many nodes. A material
# emits its AOVs correctly at 3796 nodes and emits nothing at 4387, so the cliff
# sits near 4000; 3200 leaves headroom for the last field admitted, which can
# itself be over a thousand nodes.
MAX_HOST_SHADER_NODES = 3200

# The measured cliff. Past this a material emits no AOVs at all, so crossing it
# is never acceptable: a scene that does is rejected rather than shipped with
# invisible labels. The budget above is the *pre*-injection gate; this is the
# post-injection assertion, because one field can be 1500 nodes on its own and
# the gate cannot see how expensive the next defect will turn out to be.
#
# 3800 is conservative against a cliff measured at 3796 working / 4387 dead on
# one GPU and Blender build. Treat it as a heuristic, not a specification -
# re-measure with tools/ if the renderer or hardware changes.
HOST_SHADER_NODE_CLIFF = 3800

# A defect claims a pixel in the instance map once it covers most of it. Matches
# the exporter's mask threshold so the instance map and the class masks agree on
# where a defect begins and ends.
INSTANCE_COVERAGE_GATE = 0.5


def reset_registry() -> None:
    _REGISTRY.clear()
    _INSTANCES.clear()
    _RUN_METADATA.clear()


# --------------------------------------------------------------------------- #
# geometry: where on the host does this defect sit
# --------------------------------------------------------------------------- #


def _world_matrix(obj: bpy.types.Object):
    """A trustworthy world matrix for a defect card.

    matrix_world is cached by the depsgraph, and defect cards live in the
    `unique_assets` collection which the view layer never evaluates, so theirs
    stays at the identity no matter how many view_layer.update() calls you make
    - the card reports itself at the world origin and every injected field lands
    off the wall. matrix_basis is recomputed from loc/rot/scale on access, so it
    is always current; it is the full answer for an unparented object, which
    these are.
    """
    if obj.parent is None:
        return obj.matrix_basis
    return obj.matrix_world


def _plane_frame(plane: bpy.types.Object):
    """(origin, tangent_u, tangent_v) in world space for a defect card.

    Taken from the card's own faces rather than its rotation, because the
    factories bake different rotations into the mesh before placement.
    """
    mesh = plane.data
    if not mesh.polygons:
        return None
    world = _world_matrix(plane)
    mat3 = world.to_3x3()
    normal = Vector((0.0, 0.0, 0.0))
    for poly in mesh.polygons:
        normal += mat3 @ poly.normal
    if normal.length < 1e-8:
        return None
    normal.normalize()
    up = Vector((0.0, 0.0, 1.0))
    if abs(normal.dot(up)) > 0.9:
        up = Vector((1.0, 0.0, 0.0))
    t_u = up.cross(normal)
    if t_u.length < 1e-8:
        return None
    t_u.normalize()
    t_v = normal.cross(t_u).normalized()
    # Centroid of the actual geometry, not matrix_world.translation: the plane
    # factories bake placement into the mesh with butil.apply_transform, so the
    # object translation is the identity and the offset lives in the vertices.
    # Using the translation centres every field on the world origin instead of
    # on the wall, and the defect renders nowhere.
    centre = Vector((0.0, 0.0, 0.0))
    for vert in mesh.vertices:
        centre += world @ vert.co
    centre /= max(len(mesh.vertices), 1)
    return centre, t_u, t_v


def _local_frame(host: bpy.types.Object, frame):
    """Express a world-space frame in the host object's local coordinates."""
    inv = host.matrix_world.inverted()
    inv3 = inv.to_3x3()
    origin, t_u, t_v = frame
    return inv @ origin, (inv3 @ t_u).normalized(), (inv3 @ t_v).normalized()


def _uv_socket(nt, origin, t_u, t_v):
    """A centred (u, v, 0) vector in metres on the host surface.

    Object coordinates minus the defect centre, projected onto the two tangents,
    so the field lands where the solver put the card whatever the wall's shape.
    """
    tex = nt.nodes.new("ShaderNodeTexCoord")
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    nt.links.new(tex.outputs["Object"], sub.inputs[0])
    sub.inputs[1].default_value = (origin.x, origin.y, origin.z)

    def dot(vec):
        d = nt.nodes.new("ShaderNodeVectorMath")
        d.operation = "DOT_PRODUCT"
        nt.links.new(sub.outputs["Vector"], d.inputs[0])
        d.inputs[1].default_value = (vec.x, vec.y, vec.z)
        return d.outputs["Value"]

    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(dot(t_u), comb.inputs["X"])
    nt.links.new(dot(t_v), comb.inputs["Y"])
    return comb.outputs["Vector"], tex


def _pattern_socket(nt, tex, offset):
    """A decorrelated 3D vector, for noise that must not follow the outline."""
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.vector_type = "POINT"
    for i in range(3):
        mapping.inputs["Location"].default_value[i] = float(offset[i])
    nt.links.new(tex.outputs["Object"], mapping.inputs["Vector"])
    return mapping.outputs["Vector"]


# --------------------------------------------------------------------------- #
# per-host material, shared by every defect landing on that host
# --------------------------------------------------------------------------- #


def _host_normals_point_outward(host: bpy.types.Object) -> bool:
    """True when Displacement along face normals moves away from the occupant.

    Ceiling and floor shells are extracted with inward winding, so a negative
    crack height already carves into the finish. Wall shells keep the
    solidified room volume's outward normals: the same negative height then
    moves toward the camera and the groove reads as a welt.
    """
    # A controlled solid wall is not a room volume: its visible outward face
    # points TOWARD the occupant. Allow the host owner to declare this explicitly
    # instead of interpreting the solid panel's centre as the room centre.
    explicit_sign = host.get("syndefect_displacement_sign")
    if explicit_sign is not None:
        if explicit_sign not in (-1, 1):
            raise ValueError("syndefect_displacement_sign must be -1 or +1")
        return explicit_sign == -1
    mw = host.matrix_world
    name = host.name
    dots = []
    if name.endswith(".ceiling"):
        room_dir = Vector((0.0, 0.0, -1.0))
        for poly in host.data.polygons:
            normal = mw.to_3x3() @ poly.normal
            if normal.length_squared < 1e-12:
                continue
            dots.append(normal.normalized().dot(room_dir))
    elif name.endswith(".floor"):
        room_dir = Vector((0.0, 0.0, 1.0))
        for poly in host.data.polygons:
            normal = mw.to_3x3() @ poly.normal
            if normal.length_squared < 1e-12:
                continue
            dots.append(normal.normalized().dot(room_dir))
    else:
        corners = [mw @ Vector(corner) for corner in host.bound_box]
        centre = sum(corners, Vector((0.0, 0.0, 0.0))) / max(len(corners), 1)
        for poly in host.data.polygons:
            normal = mw.to_3x3() @ poly.normal
            if normal.length_squared < 1e-12:
                continue
            normal.normalize()
            if abs(normal.z) > 0.7:
                continue
            to_room = centre - (mw @ poly.center)
            to_room.z = 0.0
            if to_room.length_squared < 1e-12:
                continue
            dots.append(normal.dot(to_room.normalized()))
    if not dots:
        return False
    return (sum(dots) / len(dots)) < 0.0


def _host_injection(host: bpy.types.Object) -> HostInjection | None:
    """Get or create the private material copy this host's defects write into.

    Room shells often share one material datablock between several walls, so
    injecting into the original would paint the same crack on all of them.
    """
    existing = _REGISTRY.get(host.name)
    if existing is not None:
        return existing
    source = next((m for m in host.data.materials if m is not None), None)
    if source is None or source.node_tree is None:
        logger.warning("host %s has no usable material; skipping", host.name)
        return None
    copy = source.copy()
    copy.name = f"{source.name}.defects"
    for i, slot_mat in enumerate(host.data.materials):
        if slot_mat is source:
            host.data.materials[i] = copy
    flip = _host_normals_point_outward(host)
    state = HostInjection(host=host, material=copy, flip_displacement=flip)
    if flip:
        logger.info(
            "host %s: flipping defect displacement (mesh normals point out of the room)",
            host.name,
        )
    _REGISTRY[host.name] = state
    return state


def _math(nt, op, a=None, b=None, clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for i, operand in enumerate((a, b)):
        if operand is None:
            continue
        if hasattr(operand, "is_linked"):
            nt.links.new(operand, n.inputs[i])
        else:
            n.inputs[i].default_value = float(operand)
    return n.outputs["Value"]


def _mix_rgb(nt, fac, c1, c2):
    n = nt.nodes.new("ShaderNodeMixRGB")
    n.blend_type = "MIX"
    for socket, val in (("Fac", fac), ("Color1", c1), ("Color2", c2)):
        if hasattr(val, "is_linked"):
            nt.links.new(val, n.inputs[socket])
        else:
            n.inputs[socket].default_value = val
    return n.outputs["Color"]


def _principled(nt):
    return next(
        (n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None
    )


def _record(
    state: HostInjection, kind: str, height_socket, label_socket, instance_id: int
) -> None:
    state.height_terms.append(height_socket)
    class_id = DEFECT_AOV_CLASSES[KIND_PREFIX[kind]][0]
    state.class_fields.setdefault(class_id, []).append(label_socket)
    state.instance_fields.append((label_socket, int(instance_id)))
    state.count += 1


# --------------------------------------------------------------------------- #
# the three defects
# --------------------------------------------------------------------------- #


@gin.configurable
def _inject_crack(
    state,
    frame,
    seed,
    extent,
    instance_id,
    crack_grime_strength=1.0,
    crack_clean_fraction=0.30,
    crack_grime_focus=2.2,
    crack_width_scale=1.0,
    crack_profile="legacy",
    crack_network_fraction=0.0,
    crack_network_spacing=(0.02, 0.06),
    crack_network_grime=(0.15, 0.60),
    crack_jitter_scale=1.0,
    crack_depth_scale=1.0,
    crack_fit_path=False,
    crack_label_mode="profile",
    crack_branch_mode="original",
    crack_path_mode="original",
    layered_fraction=0.0,
    crack_render="geometry",
):
    """Inject one crack. Opt-in: layered_fraction > 0 gives that share of cracks
    a two-layer look (see below).

    crack_render (ablation of how the crack is built, labels identical):
      "geometry" - groove, chips, lips and offset are real displacement (default);
      "pbr"      - the same height drives a bump/normal map only, like a PBR decal;
      "albedo"   - colour only (grime, exposed core), flat: a pasted crack image
                   that is still lit by the scene but has no relief at all.
    """
    if crack_render not in ("geometry", "pbr", "albedo"):
        raise ValueError("crack_render must be geometry, pbr or albedo")
    nt = state.material.node_tree
    from cwg_infinigen.assets.defect_fields import adapt_crack_branches, adapt_crack_morphology, adapt_crack_path

    from cwg_infinigen.assets.defect_fields import sample_crack_network
    from cwg_infinigen.core.util.math import int_hash
    if not 0 <= crack_network_fraction <= 1:
        raise ValueError("network fraction must be 0..1")
    network_rng = np.random.default_rng(int_hash((int(seed), "network_selection_v1")))
    network = crack_profile == "network" or network_rng.random() < crack_network_fraction
    half = min(extent, 0.42) * 0.42
    p = (sample_crack_network(seed, half, crack_network_spacing, crack_network_grime) if network else
         sample_crack_morphology(seed, half=half, profile=crack_profile))
    p = adapt_crack_morphology(
        p, width_scale=crack_width_scale, jitter_scale=crack_jitter_scale,
        depth_scale=crack_depth_scale, fit_path=crack_fit_path and not network,
    )
    if not network:
        p = adapt_crack_path(p, seed, crack_path_mode)
        p = adapt_crack_branches(p, crack_branch_mode)
    uv, _tex = _uv_socket(nt, *frame)
    geo = {}
    prof = build_crack_field(nt, uv, p, out=geo)
    height = _math(nt, "MULTIPLY", prof, -float(p["crack_depth"]))

    # Coating edge, opt-in per crack and only on hosts that have one.
    # Paint and plaster: the film tears, with lips and chips a few millimetres
    # wide. Stucco: the finish coat breaks, with no lifted film. Brick, stone,
    # concrete, tile and the other single-material hosts keep the groove alone.
    # Chips stay labelled crack, matching the BD3 paint polygons
    # (e.g. minor_crack_0401/0304).
    gen = state.host.get("cwg_material_generator")
    lay = sample_crack_layer(seed, crack_layer_mode(gen, state.host.get("surface_finish")), layered_fraction)
    if network:
        lay = None  # Crazing is shallow: no broad chips or one-sided wall step.
    chips = None
    if lay is not None:
        from cwg_infinigen.assets.defect_fields import _nodes

        _, _, mrange, noise = _nodes(nt)
        wide = dict(p, w_min=float(p["w_min"]) + lay["margin"], w_max=float(p["w_max"]) + lay["margin"])
        band_prof = build_crack_field(nt, uv, wide)
        band = mrange(band_prof, 0.0, 0.03, 0.0, 1.0)
        groove = mrange(prof, 0.0, 0.02, 0.0, 1.0)
        outside = _math(nt, "SUBTRACT", 1.0, groove)
        blob = noise(uv, lay["chip_scale"], 2.0, 0.5)
        chips = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", mrange(blob, lay["chip_thr"], lay["chip_thr"] + 0.02),
                                            band, clamp=True), outside, clamp=True)
        runs = mrange(noise(uv, lay["chip_scale"] * 0.25, 2.0, 0.5), 0.45, 0.60)
        lifted = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", mrange(band_prof, 0.02, 0.30, 0.0, 1.0, "SMOOTHSTEP"),
                                             outside, clamp=True),
                       _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", 1.0, chips), runs, clamp=True), clamp=True)
        height = _math(nt, "ADD", height, _math(nt, "SUBTRACT", _math(nt, "MULTIPLY", lifted, lay["lip"]),
                                                _math(nt, "MULTIPLY", chips, lay["film"])))
        if lay["offset"] > 0.0:
            # One side proud of the other, fading out toward the crack's tips
            # and the card border so the step never runs off past the crack.
            signed = _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", geo["u"], geo["path"]), lay["side"])
            side = mrange(signed, -0.001, 0.001, 0.0, 1.0, "SMOOTHSTEP")
            half_v = float(p.get("half_v", 0.14))
            half_u = float(p.get("half_u", 0.14))
            along = mrange(_math(nt, "ABSOLUTE", geo["v"]), half_v * 0.55, half_v * 0.95, 1.0, 0.0, "SMOOTHSTEP")
            across = mrange(_math(nt, "ABSOLUTE", geo["u"]), half_u * 0.60, half_u * 0.95, 1.0, 0.0, "SMOOTHSTEP")
            step = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", side, along, clamp=True), across, clamp=True)
            height = _math(nt, "ADD", height, _math(nt, "MULTIPLY", step, lay["offset"]))
        bsdf_ = _principled(nt)
        if bsdf_ is not None and bsdf_.inputs["Base Color"].is_linked:
            src_ = bsdf_.inputs["Base Color"].links[0].from_socket
            # The groove keeps the parent material. A fixed plaster/cement/putty
            # colour read as a white stripe on brick, stone and mid paint, and
            # it did not change with the wall. Darken the colour already in the
            # shader; the displacement supplies the shadow.
            exposed = _math(nt, "MAXIMUM", mrange(prof, 0.02, 0.5, 0.0, 1.0), chips, clamp=True)
            dark = nt.nodes.new("ShaderNodeMixRGB")
            dark.blend_type = "MULTIPLY"
            dark.inputs["Fac"].default_value = 1.0
            nt.links.new(src_, dark.inputs["Color1"])
            dark.inputs["Color2"].default_value = (lay["dark"],) * 3 + (1.0,)
            nt.links.new(_mix_rgb(nt, exposed, src_, dark.outputs["Color"]), bsdf_.inputs["Base Color"])
            lay["painted_host"] = lay.get("mode") == "paint"

    # Grime in the channel, sampled per crack rather than applied to all of them.
    #
    # Real cracks collect dust and show darker substrate; a perfectly clean crack
    # in fresh paint is the exception. But grime is the one part of a crack that
    # does NOT depend on where the light is, so a constant strength on every
    # instance is what made the old version read as a painted-on line and what
    # let a defect stay visible where a real one would wash out.
    #
    # crack_clean_fraction of cracks get none at all. Those are the instances
    # that carry the light-response claim on their geometry alone; the rest look
    # like cracks in a real building. Report the lighting sweep on the clean
    # subset - it is a conditional claim, and a stronger one than a blanket
    # average over instances that were partly pigment all along.
    #
    # LABELS ARE UNAFFECTED EITHER WAY. The AOV is fed by `prof`; nothing here
    # touches it. A crack that renders invisible is still labelled exactly as
    # confidently as a visible one, which is the intended contract for all three
    # classes - see _wire_labels.
    #
    # Derived from seed arithmetic, never from np.random: this function's RNG
    # stream is compared draw-for-draw against the card factory by
    # tests/test_defect_field_drift.py, so consuming a draw would break the seed
    # contract. sampled_value has always been seed arithmetic for this reason.
    # int_hash, not seed arithmetic, for the clean/grimy split. `(seed // 7) %
    # 100` looked like a hash and is not one: over 400 consecutive seeds it put
    # 52% of cracks in the clean bucket against a requested 30%, because seed//7
    # only spans 0..57 before the modulo ever bites. A requested fraction that
    # silently comes out at 52% is exactly the kind of thing that makes a
    # dataset statistic wrong without anything failing.
    from cwg_infinigen.core.util.math import int_hash

    clean_cut = int(round(max(0.0, min(1.0, float(crack_clean_fraction))) * 1000))
    grimy = (int_hash(("crack_grime", int(seed))) % 1000) >= clean_cut
    strength = float(crack_grime_strength) if grimy else 0.0
    if network:
        # Shallow crazing needs weak pigment evidence even in the clean bucket.
        # A zero crack_grime_strength still explicitly disables it for ablations.
        strength = float(crack_grime_strength) * p["network"]["grime_strength"]

    bsdf = _principled(nt)
    if network and bsdf is not None and not bsdf.inputs["Base Color"].is_linked:
        # Controlled paint uses a constant color instead of a texture link.
        base = nt.nodes.new("ShaderNodeRGB")
        base.outputs["Color"].default_value = bsdf.inputs["Base Color"].default_value
        nt.links.new(base.outputs["Color"], bsdf.inputs["Base Color"])
    if strength > 0.0 and bsdf is not None and bsdf.inputs["Base Color"].is_linked:
        src = bsdf.inputs["Base Color"].links[0].from_socket
        hsv = nt.nodes.new("ShaderNodeHueSaturation")
        # 0.42-0.70 of the wall's value. The old range topped out at 0.92, which
        # is 4 code values of contrast at correct exposure - invisible, and it
        # still cost the nodes. Dropped.
        sampled_value = float(np.clip(0.42 + 0.28 * ((seed % 7) / 6.0), 0.42, 0.70))
        hsv.inputs["Value"].default_value = 1.0 - (1.0 - sampled_value) * strength
        nt.links.new(src, hsv.inputs["Color"])
        # prof ** focus, not prof: grime settles in the deepest part of the
        # channel. Mixing by the raw profile darkens the full width evenly,
        # which is a band of paint rather than dirt in a groove.
        fac = _math(nt, "POWER", prof, float(crack_grime_focus))
        nt.links.new(
            _mix_rgb(nt, fac, src, hsv.outputs["Color"]), bsdf.inputs["Base Color"]
        )
    if crack_label_mode not in ("profile", "support"):
        raise ValueError("crack_label_mode must be profile or support")
    # prof is normalized groove depth, NOT binary surface coverage. Passing it
    # straight to an AOV and thresholding at .5 labels only the deepest half of
    # a V-channel. For the explicit support policy, each shader sample labels
    # the channel footprint; Cycles averaging supplies pixel antialias coverage.
    # The .02 floor excludes the vanishing border fade, in field units, not px.
    label = prof if crack_label_mode == "profile" else _math(nt, "GREATER_THAN", prof, .02)
    if chips is not None:
        label = _math(nt, "MAXIMUM", label, chips, clamp=True)  # edge chips are part of the crack
    if crack_render == "pbr":
        # The same metric height as a normal map: shading changes with the light,
        # but nothing moves, so no silhouette, self-shadow or occlusion.
        bsdf_b = _principled(nt)
        if bsdf_b is not None:
            bump = nt.nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = 1.0
            bump.inputs["Distance"].default_value = -1.0 if state.flip_displacement else 1.0
            nt.links.new(height, bump.inputs["Height"])
            nrm = bsdf_b.inputs["Normal"]
            if nrm.is_linked:
                src_n = nrm.links[0].from_socket
                nt.links.new(src_n, bump.inputs["Normal"])
            nt.links.new(bump.outputs["Normal"], nrm)
    if crack_render != "geometry":
        height = _math(nt, "MULTIPLY", height, 0.0)  # no real displacement from the crack
    _record(state, "crack", height, label, instance_id)
    # Metric morphology, distinct from the card that carries it. The card is a
    # placement token; these are the defect's own dimensions, and without them
    # `extent_m` is the only size in the manifest and it describes the wrong
    # thing entirely.
    return p["style"], {
        "style": p["style"],
        "crack_width_min_m": round(float(p["w_min"]), 6),
        "crack_width_max_m": round(float(p["w_max"]), 6),
        "crack_depth_m": round(float(p["crack_depth"]), 6),
        # Historical crack_width_* fields are channel radii, retained for compatibility.
        "width_convention": ("channel_radius_before_coordinate_warp" if network else
                             "channel_radius_before_tip_taper"),
        "full_opening_min_m": round(2 * float(p["w_min"]), 6),
        "full_opening_max_m": round(2 * float(p["w_max"]), 6),
        "profile": crack_profile,
        "network": p.get("network"),
        "grime_strength_effective": strength,
        "width_scale": float(crack_width_scale),
        "jitter_scale": float(crack_jitter_scale),
        "depth_scale": float(crack_depth_scale),
        "fit_path": bool(crack_fit_path and not network),
        "branch_mode": "cellular" if network else crack_branch_mode,
        "branch_specs": p["fork_specs"],
        "path_mode": "cellular" if network else crack_path_mode,
        "path_knots_m": p.get("path_knots"),
        "label_mode": crack_label_mode,
        "render": crack_render,
        "label_support_profile_floor": .02 if crack_label_mode == "support" else None,
        "lip_width_m": round(float(p.get("w_lip", 0.0)), 6),
        "lip_height_m": round(float(p.get("h_lip", 0.0)), 6),
        "n_forks": len(p.get("fork_specs") or ()),
        "n_kinks": len(p.get("kink_specs") or ()),
        "layered": None if lay is None else {
            "mode": lay.get("mode"),
            "core": lay["core"], "margin_m": round(lay["margin"], 6), "film_m": round(lay["film"], 6),
            "lip_m": round(lay["lip"], 6), "offset_m": round(lay["offset"], 6),
            "painted_host": lay.get("painted_host")},
    }


_INJECTORS = {
    "crack": _inject_crack,
}


def inject_defect(
    plane: bpy.types.Object,
    host: bpy.types.Object,
    kind: str,
    seed: int,
    instance_id: int = 0,
) -> bool:
    """Rebuild one solver-placed defect inside its host's material.

    The plane is not modified; the caller decides whether to delete it.
    """
    if kind not in _INJECTORS:
        plane["syndefect_injection_error"] = "unsupported_kind"
        return False
    frame_w = _plane_frame(plane)
    if frame_w is None:
        logger.warning("defect %s has no usable faces; skipping", plane.name)
        plane["syndefect_injection_error"] = "invalid_plane_frame"
        return False
    state = _host_injection(host)
    if state is None:
        plane["syndefect_injection_error"] = "host_has_no_usable_material"
        return False
    extent = max(plane.dimensions.x, plane.dimensions.y, plane.dimensions.z)
    try:
        result = _INJECTORS[kind](
            state, _local_frame(host, frame_w), seed, float(extent), instance_id
        )
        style, metrics = result if isinstance(result, tuple) else (result, {})
    except Exception as exc:
        logger.warning(
            "failed to inject %s %s into %s: %s", kind, plane.name, host.name, exc
        )
        plane["syndefect_injection_error"] = f"injection_exception:{type(exc).__name__}"
        return False
    logger.debug("injected %s (%s, seed %d) into %s", kind, style, seed, host.name)
    plane["syndefect_style"] = str(style)
    if metrics:
        plane["syndefect_morphology"] = json.dumps(metrics)
    if "syndefect_injection_error" in plane:
        del plane["syndefect_injection_error"]
    return True


def _wire_displacement(state: HostInjection) -> None:
    """Add every defect's height into the host's own displacement chain.

    The plaster carries its orange-peel on the Material Output's Displacement
    socket, so replacing that link would flatten the wall. Height is metres and
    the node's Scale divides it back out.
    """
    nt = state.material.node_tree
    out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    if out is None:
        return
    total = state.height_terms[0]
    for term in state.height_terms[1:]:
        total = _math(nt, "ADD", total, term)
    # Injectors use occupant space: positive proud, negative carved in.
    # Wall shells displace along outward geometric normals, so that convention
    # has to be negated here or a crack comes toward the camera as a ridge.
    if state.flip_displacement:
        total = _math(nt, "MULTIPLY", total, -1.0)

    disp = (
        out.inputs["Displacement"].links[0].from_node
        if out.inputs["Displacement"].is_linked
        else None
    )
    if disp is not None and disp.bl_idname == "ShaderNodeDisplacement":
        height_in = disp.inputs["Height"]
        scale = disp.inputs["Scale"].default_value or 1.0
        if height_in.is_linked:
            host_height = height_in.links[0].from_socket
            nt.links.remove(height_in.links[0])
            nt.links.new(
                _math(nt, "ADD", host_height, _math(nt, "DIVIDE", total, scale)),
                height_in,
            )
        else:
            nt.links.new(
                _math(nt, "ADD", disp.inputs["Midlevel"].default_value, total),
                height_in,
            )
    else:
        disp = nt.nodes.new("ShaderNodeDisplacement")
        disp.inputs["Midlevel"].default_value = 0.0
        disp.inputs["Scale"].default_value = 1.0
        nt.links.new(total, disp.inputs["Height"])
        for link in list(out.inputs["Displacement"].links):
            nt.links.remove(link)
        nt.links.new(disp.outputs["Displacement"], out.inputs["Displacement"])
    state.material.displacement_method = "DISPLACEMENT"


@gin.configurable
def _wire_labels(state: HostInjection) -> None:
    """One coverage AOV per class, plus one AOV carrying instance identity."""
    nt = state.material.node_tree
    id_to_aov = {cid: aov for (cid, aov) in DEFECT_AOV_CLASSES.values()}
    merged_by = {}
    for class_id, fields in state.class_fields.items():
        merged = fields[0]
        for extra in fields[1:]:
            merged = _math(nt, "MAXIMUM", merged, extra, clamp=True)
        merged_by[class_id] = merged
    for class_id, merged in merged_by.items():
        aov = nt.nodes.new("ShaderNodeOutputAOV")
        aov.aov_name = id_to_aov[class_id]
        nt.links.new(merged, aov.inputs["Value"])

    _wire_instance_label(state)


def _wire_instance_label(state: HostInjection) -> None:
    """Write coverage plus the first two moments of the winning instance id.

    The class AOVs cannot separate two cracks on one wall: they are merged with
    MAXIMUM, so the union is all that survives. This adds one more VALUE AOV
    carrying the instance id instead of coverage, which is what makes instance
    masks, per-instance recall and per-instance metrics possible.

    The id is gated on coverage rather than multiplied by it. Scaling the id by
    a fractional coverage and taking the maximum would let a barely-covered
    high id outrank a fully-covered low one - id 5 at coverage 0.3 beats id 1 at
    coverage 1.0 - and decode to the wrong instance. Gating means a defect
    contributes its id only where it actually owns the pixel, and where two
    defects both do, the higher id wins deterministically.

    Cycles averages AOV values across samples.  The three outputs let the
    exporter recover ``E[id] / coverage`` at an antialiased edge and calculate
    the conditional variance.  A non-zero variance means multiple ids
    contributed to that pixel, which is explicitly ignored at training time.
    All three are written to EXR by the compositor.
    """
    if not state.instance_fields:
        return
    nt = state.material.node_tree
    merged = None
    for coverage, instance_id in state.instance_fields:
        owns = _math(nt, "GREATER_THAN", coverage, INSTANCE_COVERAGE_GATE)
        tagged = _math(nt, "MULTIPLY", owns, float(instance_id))
        merged = tagged if merged is None else _math(nt, "MAXIMUM", merged, tagged)
    occupied = _math(nt, "GREATER_THAN", merged, 0.0)
    squared = _math(nt, "MULTIPLY", merged, merged)
    for aov_name, value in (
        (DEFECT_INSTANCE_AOV, merged),
        (DEFECT_INSTANCE_COVERAGE_AOV, occupied),
        (DEFECT_INSTANCE_SQUARED_AOV, squared),
    ):
        aov = nt.nodes.new("ShaderNodeOutputAOV")
        aov.aov_name = aov_name
        nt.links.new(value, aov.inputs["Value"])


@gin.configurable
def finalize_injections(
    dicing_rate: float = 2.0, enable_subdivision: bool = True
) -> int:
    """Wire up every host that received defects, and turn on real displacement.

    dicing_rate is in pixels. Measured on a five-room scene at 1280x720, going
    from 2.0 to 0.5 costs 19x the peak memory (2.0 GB -> 38.5 GB) and changes a
    crack's measured contrast by under 5%, so the default is deliberately coarse.
    """
    if not _REGISTRY:
        return 0
    scene = bpy.context.scene
    if enable_subdivision:
        scene.cycles.feature_set = "EXPERIMENTAL"
        scene.cycles.dicing_rate = dicing_rate
    total = 0
    host_reports = []
    for state in _REGISTRY.values():
        if not state.height_terms:
            continue
        _wire_displacement(state)
        _wire_labels(state)
        if enable_subdivision:
            host = state.host
            if not any(m.type == "SUBSURF" for m in host.modifiers):
                mod = host.modifiers.new("defect_adaptive", "SUBSURF")
                mod.subdivision_type = "SIMPLE"
            host.cycles.use_adaptive_subdivision = True
            host.cycles.dicing_rate = dicing_rate
        total += state.count
        host_reports.append(
            {
                "host_object": state.host.name,
                "injected": state.count,
                "shader_node_count": len(state.material.node_tree.nodes),
                "material": state.material.name,
            }
        )
        logger.info(
            "host %s: %d defect(s) injected into %s",
            state.host.name,
            state.count,
            state.material.name,
        )
    _RUN_METADATA["hosts"] = sorted(host_reports, key=lambda item: item["host_object"])
    return total


# --------------------------------------------------------------------------- #
# sweep: find the placed defects, inject them, remove the cards
# --------------------------------------------------------------------------- #

_PREFIX_KIND = {prefix: kind for kind, prefix in KIND_PREFIX.items()}

_SURFACE_SUFFIX = {
    "wall": ".wall",
    "ceiling": ".ceiling",
    "floor": ".floor",
}

# Paint film can peel or blister only where a coating exists.  Cracks are
# supported on common brittle finishes but not on timber/metal/glass cards.
# Peel and blistering are *coating* failures: they need a coating to fail, so a
# bare timber or tile floor cannot have them and a defect claiming otherwise is
# not a defect, it is a mistake.
_COATED_FINISHES = {"paint", "coated", "plaster", "epoxy", "painted_concrete"}
# Cracking is a *substrate* failure and every indoor substrate does it - timber
# splits, tiles crack, stone, concrete, masonry and plaster all crack. None is
# listed here because the answer is "any": injection folds the crack field into a
# copy of the host's own material and the grime tint is drawn from that host's
# albedo, so a crack in a timber floor is already rendered in the timber's own
# colour rather than in plaster's.
_CRACK_FINISHES = None


def classify_plane(obj: bpy.types.Object) -> str | None:
    """Which injectable defect this object is, by material name."""
    if obj.type != "MESH" or obj.data is None:
        return None
    for material in obj.data.materials:
        if material is None:
            continue
        for prefix, kind in _PREFIX_KIND.items():
            if (material.name or "").startswith(prefix):
                return kind
    return None


def _stored_morphology(plane) -> dict | None:
    """The defect's own sampled dimensions, as stashed at injection time.

    `extent_m` is the placement card and says nothing about the defect inside
    it - a metre-wide card can carry a 0.4 mm crack. These are the numbers any
    claim about defect size has to rest on, and the ones a measurement-error
    metric would be scored against.
    """
    raw = plane.get("syndefect_morphology")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _host_material_node_count(host) -> int:
    """Nodes in the host's injected material, or 0 before the first injection."""
    state = _REGISTRY.get(getattr(host, "name", ""))
    if state is None or state.material is None or state.material.node_tree is None:
        return 0
    return len(state.material.node_tree.nodes)


def classify_host_finish(host) -> str:
    """Return a stable substrate category used by placement QA.

    Room material assignment stamps ``surface_finish``.  Name inspection is a
    compatibility fallback for older saved scenes, and ``unknown`` remains
    explicit in the manifest instead of being silently called paint.
    """
    explicit = str(host.get("surface_finish", "")).strip().lower()
    if explicit:
        return explicit
    names = [host.name]
    names.extend(m.name for m in getattr(host.data, "materials", []) if m is not None)
    signature = " ".join(names).lower()
    for finish, tokens in (
        ("paint", ("plaster", "paint", "coated", "epoxy")),
        ("concrete", ("concrete", "cement", "screed")),
        ("masonry", ("brick", "masonry")),
        ("tile", ("tile", "ceramic")),
        ("timber", ("wood", "parquet", "hardwood")),
        ("stone", ("stone", "marble", "granite")),
        ("metal", ("metal", "steel", "aluminium", "aluminum")),
        ("glass", ("glass",)),
    ):
        if any(token in signature for token in tokens):
            return finish
    return "unknown"


def substrate_compatibility(
    kind: str, surface: str | None, host
) -> tuple[bool, str, str | None]:
    """Whether this defect has a physically meaningful host coating."""
    finish = classify_host_finish(host)
    allowed = _CRACK_FINISHES if kind == "crack" else _COATED_FINISHES
    if allowed is None or finish in allowed:
        return True, finish, None
    # Older wall/ceiling scenes predate surface_finish but are overwhelmingly
    # plaster.  Unknown floors are still refused for coating failures, because an
    # unknown floor may well be bare wood or tile; cracks no longer reach this
    # branch at all, since every substrate cracks.
    if finish == "unknown" and surface in {"wall", "ceiling"}:
        return True, finish, "legacy_unknown_finish"
    return False, finish, f"incompatible_{finish}_substrate"


def validate_defect_support(
    plane,
    host,
    distance_tolerance_m: float = 0.03,
    normal_tolerance_deg: float = 20.0,
    require_named_surface: bool = True,
) -> dict:
    """Conservatively check the complete card footprint against its host.

    The eight object-space bounding-box corners contain the full defect card,
    unlike a centre-only test.  Every corner must resolve onto the intended
    host within the metric tolerance, and the card and host normals must be
    parallel.  Normal sign is reported separately because old room meshes do
    not consistently orient their winding toward the room interior.
    """
    surface = plane.get("syndefect_surface")
    expected_suffix = _SURFACE_SUFFIX.get(surface)
    reasons = []
    if (
        require_named_surface
        and expected_suffix
        and not host.name.endswith(expected_suffix)
    ):
        reasons.append("host_surface_mismatch")

    frame = _plane_frame(plane)
    if frame is None:
        return {
            "valid": False,
            "reasons": reasons + ["invalid_plane_frame"],
            "distance_tolerance_m": float(distance_tolerance_m),
            "normal_tolerance_deg": float(normal_tolerance_deg),
        }

    _, t_u, t_v = frame
    centre = frame[0]
    plane_normal = t_u.cross(t_v).normalized()
    plane_world = _world_matrix(plane)
    host_inverse = host.matrix_world.inverted()
    normal_matrix = host.matrix_world.to_3x3().inverted().transposed()
    distances = []
    signed_alignments = []
    unresolved = 0
    for corner in plane.bound_box:
        world_point = plane_world @ Vector(corner)
        found, closest, normal, _ = host.closest_point_on_mesh(
            host_inverse @ world_point
        )
        if not found:
            unresolved += 1
            continue
        closest_world = host.matrix_world @ closest
        host_normal = normal_matrix @ normal
        if host_normal.length < 1e-8:
            unresolved += 1
            continue
        host_normal.normalize()
        distances.append(float((world_point - closest_world).length))
        signed_alignments.append(float(plane_normal.dot(host_normal)))

    max_distance = max(distances, default=float("inf"))
    min_abs_alignment = min((abs(v) for v in signed_alignments), default=0.0)
    required_alignment = math.cos(math.radians(normal_tolerance_deg))
    room_direction = None
    if surface == "floor":
        room_direction = Vector((0.0, 0.0, 1.0))
    elif surface == "ceiling":
        room_direction = Vector((0.0, 0.0, -1.0))
    elif surface == "wall":
        host_corners = [host.matrix_world @ Vector(corner) for corner in host.bound_box]
        host_centre = sum(host_corners, Vector((0.0, 0.0, 0.0))) / max(
            len(host_corners), 1
        )
        room_direction = host_centre - centre
        room_direction.z = 0.0
        if room_direction.length > 1e-8:
            room_direction.normalize()
        else:
            room_direction = None
    room_facing_dot = (
        float(plane_normal.dot(room_direction)) if room_direction is not None else None
    )
    if unresolved:
        reasons.append("unresolved_support_corner")
    if max_distance > distance_tolerance_m:
        reasons.append("footprint_outside_host")
    if min_abs_alignment < required_alignment:
        reasons.append("normal_misaligned")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "corners_checked": len(plane.bound_box),
        "unresolved_corners": unresolved,
        "max_support_distance_m": round(max_distance, 6),
        "min_abs_normal_dot": round(min_abs_alignment, 6),
        "normal_sign": (
            "same" if signed_alignments and sum(signed_alignments) >= 0 else "opposite"
        ),
        "room_facing_normal_dot": (
            round(room_facing_dot, 6) if room_facing_dot is not None else None
        ),
        "distance_tolerance_m": float(distance_tolerance_m),
        "normal_tolerance_deg": float(normal_tolerance_deg),
    }


def resolve_host(plane, state, host_by_name):
    """The wall or ceiling this defect was placed against.

    Prefers the solver relation that put it there; falls back to the nearest
    host, which is what the plane factories do for their colour lookup.
    """
    surface = plane.get("syndefect_surface")
    order = {
        "ceiling": (".ceiling", ".wall", ".floor"),
        "floor": (".floor", ".wall", ".ceiling"),
    }
    suffixes = order.get(surface, (".wall", ".ceiling", ".floor"))
    if state is not None and host_by_name:
        from cwg_infinigen.core import tags as t

        for objstate in state.objs.values():
            if objstate.obj is not plane:
                continue
            for rel in objstate.relations:
                room_name = rel.target_name
                if (
                    room_name in state.objs
                    and t.Semantics.Room in state.objs[room_name].tags
                ):
                    stem = room_name.split(".")[0]
                    for suffix in suffixes:
                        if stem + suffix in host_by_name:
                            plane["syndefect_host_resolution"] = "solver_relation"
                            return host_by_name[stem + suffix]
            break

    frame = _plane_frame(plane)
    centre = frame[0] if frame is not None else _world_matrix(plane).translation
    best, nearest = None, None
    for host in host_by_name.values():
        d2 = (centre - host.matrix_world.translation).length_squared
        if best is None or d2 < best:
            best, nearest = d2, host
    if nearest is not None:
        plane["syndefect_host_resolution"] = "nearest_fallback"
    return nearest


def _retarget_orphaned_defect_cameras(dropped_names, survivors) -> int:
    """Re-aim close-up rigs whose defect was dropped, or disable them.

    Defect-focus rigs are posed before injection runs, so a rig can be aimed at
    a defect that the per-host cap later hides. The rig then photographs a blank
    wall: a close-up frame with no defect in it and no label, which looks like a
    labelling failure and is really a camera pointing at nothing. Re-aiming onto
    a surviving defect keeps the close-up coverage the rig was created for.
    """
    if not dropped_names:
        return 0
    pool = [p for p in survivors if p.name not in dropped_names]
    moved = 0
    for rig in bpy.data.objects:
        target_name = rig.get("syndefect_target")
        if target_name is None or target_name not in dropped_names:
            continue
        if not pool:
            rig["syndefect_orphaned"] = True
            logger.warning(
                "%s framed %s, which was dropped, and no injected defect is "
                "available to re-aim it at",
                rig.name,
                target_name,
            )
            continue
        # Round-robin through the survivors so several orphans do not all pile
        # onto the same defect.
        replacement = pool[moved % len(pool)]
        frame = _plane_frame(replacement)
        if frame is None:
            continue
        centre, t_u, t_v = frame
        normal = t_u.cross(t_v).normalized()
        offset = float(rig.get("syndefect_distance", 0.9))
        if (rig.matrix_world.translation - centre).dot(normal) < 0:
            normal = -normal
        rig.location = centre + normal * offset
        rig.rotation_euler = (
            (centre - rig.location).normalized().to_track_quat("-Z", "Y").to_euler()
        )
        rig["syndefect_target"] = replacement.name
        rig["syndefect_retargeted_from"] = target_name
        moved += 1
    if moved:
        logger.info("re-aimed %d defect camera(s) off dropped defects", moved)
    return moved


def _mark_orphaned_defect_cameras(dropped_names) -> int:
    """Keep dropped-target views as explicit negatives without moving cameras."""
    marked = 0
    for rig in bpy.data.objects:
        target_name = rig.get("syndefect_target")
        if target_name not in dropped_names:
            continue
        rig["syndefect_orphaned"] = True
        rig["syndefect_orphan_reason"] = "target_not_retained"
        marked += 1
    if marked:
        logger.info("marked %d dropped-target camera rig(s) as negative views", marked)
    return marked


def _qa_camera_records(plane_name: str) -> list[dict]:
    views = []
    for rig in sorted(bpy.data.objects, key=lambda obj: obj.name):
        if rig.get("syndefect_target") != plane_name:
            continue
        views.append(
            {
                "rig": rig.name,
                "pose_valid": bool(rig.get("syndefect_pose_valid", False)),
                "pose_reason": rig.get("syndefect_pose_reason"),
                "target_point": list(rig.get("syndefect_target_point", [])),
            }
        )
    return views


def _set_disposition(plane, status: str, reason: str | None = None) -> None:
    """Stamp a render-safe disposition onto a solver placeholder."""
    plane["syndefect_status"] = status
    if reason:
        plane["syndefect_status_reason"] = reason
    if status != "retained":
        plane.hide_render = True
        plane.hide_viewport = True
        plane["syndefect_dropped"] = True


def _record_instance(
    plane,
    host,
    kind,
    instance_id: int,
    render_path: str,
    status: str,
    reason: str | None = None,
    support: dict | None = None,
) -> dict:
    """Capture one defect's provenance, and stamp its id onto the object.

    The id is written to the object as well as the manifest so it survives into
    the saved blend: the render task and any later tooling can recover it
    without needing the manifest file alongside.
    """
    from cwg_infinigen.core.util.math import int_hash

    plane["syndefect_instance_id"] = int(instance_id)
    plane["syndefect_render_path"] = render_path
    _set_disposition(plane, status, reason)
    # Only the blister factories set this themselves; stamp it for every kind so
    # render-time tooling does not have to re-derive it from material names.
    plane["syndefect_kind"] = kind

    class_id, aov_name = DEFECT_AOV_CLASSES[KIND_PREFIX[kind]]
    frame = _plane_frame(plane)
    record = {
        "instance_id": int(instance_id),
        "kind": kind,
        "class_id": int(class_id),
        "aov_name": aov_name,
        "render_path": render_path,
        "status": status,
        "retained": status == "retained",
        "status_reason": reason,
        "factory": plane.name.split("(")[0],
        "object": plane.name,
        "seed": int(int_hash(plane.name)),
        "style": plane.get("syndefect_style"),
        "host_object": host.name if host else None,
        "host_surface": plane.get("syndefect_surface"),
        "host_material": None,
        "host_finish": classify_host_finish(host) if host else None,
        "host_resolution": plane.get("syndefect_host_resolution"),
        "extent_m": [round(float(v), 5) for v in plane.dimensions],
        "morphology": _stored_morphology(plane),
        "support_validation": support,
        "qa_views": _qa_camera_records(plane.name),
    }
    state = _REGISTRY.get(host.name) if host else None
    if state is not None:
        record["host_material"] = state.material.name
    if frame is not None:
        centre, t_u, t_v = frame
        normal = t_u.cross(t_v).normalized()
        record["world_position"] = [round(float(v), 5) for v in centre]
        record["local_frame"] = {
            "tangent_u": [round(float(v), 5) for v in t_u],
            "tangent_v": [round(float(v), 5) for v in t_v],
            "normal": [round(float(v), 5) for v in normal],
        }
    _INSTANCES.append(record)
    return record


def _summary_counts(records: list[dict]) -> dict:
    status = Counter(record["status"] for record in records)
    return {
        "sampled": len(records),
        "injected": sum(record["render_path"] == "injected" for record in records),
        "failed": status["failed"],
        "dropped": status["dropped"],
        "retained": status["retained"],
    }


def _drop_bias(records: list[dict]) -> dict:
    """Why instances were lost, grouped so the survivors' bias is visible.

    Counts alone hide the shape of the surviving population. Every filter is
    selective in a different direction - the substrate rule removes coating
    failures from bare floors, the shader budget removes whatever happens to be
    injected last on a crowded host, the QA-view rule removes defects in places a
    camera cannot reach - so "12 of 18 retained" can describe a balanced sample
    or a dataset with no floor peel in it at all. Anyone reporting class balance
    needs this breakdown rather than the totals.
    """
    bias: dict[str, dict] = {}
    for record in records:
        if record.get("status") == "retained":
            continue
        reason = str(record.get("status_reason") or "unspecified").split(":")[0]
        entry = bias.setdefault(reason, {"count": 0, "by_kind": {}, "by_surface": {}})
        entry["count"] += 1
        kind = str(record.get("kind") or "unknown")
        surface = str(record.get("host_surface") or "unknown")
        entry["by_kind"][kind] = entry["by_kind"].get(kind, 0) + 1
        entry["by_surface"][surface] = entry["by_surface"].get(surface, 0) + 1
    return {key: bias[key] for key in sorted(bias)}


def _grouped_summary(records: list[dict], field_name: str) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record.get(field_name) or "unknown")].append(record)
    return {key: _summary_counts(grouped[key]) for key in sorted(grouped)}


def write_defect_manifest(path, scene_id=None, scene_seed=None) -> int:
    """Write one JSON record per placed defect. Returns how many were written.

    This is the per-instance provenance the class AOVs cannot supply: which
    defect an instance id refers to, what made it, where it sits, how big it is,
    and whether it was injected into the host surface or left as a card. It is
    written at coarse time because that is when the information exists - the
    render task sees only materials and pixels.
    """
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit lineage. Every view, resample and rendering variant descended
    # from one coarse solve carries this id, so a split can group by the scene
    # that actually produced a frame instead of inferring it from a file path.
    # Inferring leaks: render each rig into its own folder and a path-based
    # rule sees one scene as many, then scatters near-duplicate frames of the
    # same room across train, val and test.
    payload = {
        "schema": "syndefect3d/defect_instances/2",
        "scene_id": str(scene_id) if scene_id is not None else None,
        "scene_seed": int(scene_seed) if scene_seed is not None else None,
        "instance_aov": DEFECT_INSTANCE_AOV,
        "instance_encoding": {
            "first_moment_aov": DEFECT_INSTANCE_AOV,
            "coverage_aov": DEFECT_INSTANCE_COVERAGE_AOV,
            "second_moment_aov": DEFECT_INSTANCE_SQUARED_AOV,
            "decode": "conditional_moments_with_ambiguity_rejection",
            "overlap_ownership": "highest_instance_id_at_each_shader_sample",
        },
        "instance_coverage_gate": INSTANCE_COVERAGE_GATE,
        "count": len(_INSTANCES),
        "summary": {
            **_summary_counts(_INSTANCES),
            "by_kind": _grouped_summary(_INSTANCES, "kind"),
            "by_surface": _grouped_summary(_INSTANCES, "host_surface"),
            "by_host": _grouped_summary(_INSTANCES, "host_object"),
            "drop_bias": _drop_bias(_INSTANCES),
        },
        "injection_run": dict(_RUN_METADATA),
        "instances": sorted(_INSTANCES, key=lambda r: r["instance_id"]),
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %d defect instance record(s) to %s", len(_INSTANCES), path)
    return len(_INSTANCES)


@gin.configurable
def run_injection(
    state=None,
    host_by_name=None,
    enabled: bool = False,
    hide_planes: bool = True,
    preview_planes_in_viewport: bool = True,
    dicing_rate: float = 2.0,
    max_per_host: int = MAX_DEFECTS_PER_HOST,
    max_host_shader_nodes: int = MAX_HOST_SHADER_NODES,
    drop_over_cap: bool = True,
    injection_only: bool = True,
    enforce_substrates: bool = True,
    validate_support: bool = True,
    allow_nearest_host_fallback: bool = False,
    support_distance_tolerance_m: float = 0.03,
    support_normal_tolerance_deg: float = 20.0,
    require_qa_view: bool = False,
    retarget_dropped_cameras: bool = False,
    fail_scene_on_injection_error: bool = True,
) -> int:
    """Inject every placed crack into its host surface.

    Returns the number of defects injected. Off by default; enable with the
    defect_injection.gin overlay.
    """
    if not enabled:
        return 0
    if injection_only and (not hide_planes or not drop_over_cap):
        raise ValueError(
            "injection_only requires hide_planes=True and drop_over_cap=True"
        )
    # finalize_assets nudges each card into its host just before this runs, and
    # Blender does not refresh matrix_world until the depsgraph is evaluated.
    # Without this the cards still report their pre-placement transform and
    # every field gets centred on the world origin instead of on the wall.
    bpy.context.view_layer.update()

    reset_registry()
    host_by_name = host_by_name or {}
    _RUN_METADATA.update(
        {
            "injection_only": bool(injection_only),
            "max_per_host": int(max_per_host),
            "max_host_shader_nodes": int(max_host_shader_nodes),
            "host_shader_node_cliff": int(HOST_SHADER_NODE_CLIFF),
            "drop_over_cap": bool(drop_over_cap),
            "enforce_substrates": bool(enforce_substrates),
            "validate_support": bool(validate_support),
            "allow_nearest_host_fallback": bool(allow_nearest_host_fallback),
            "support_distance_tolerance_m": float(support_distance_tolerance_m),
            "support_normal_tolerance_deg": float(support_normal_tolerance_deg),
            "require_qa_view": bool(require_qa_view),
            "retarget_dropped_cameras": bool(retarget_dropped_cameras),
            "fail_scene_on_injection_error": bool(fail_scene_on_injection_error),
        }
    )

    from cwg_infinigen.core.util.math import int_hash

    planes = []
    for obj in sorted(bpy.data.objects, key=lambda o: o.name):
        kind = classify_plane(obj)
        if kind is not None:
            planes.append((obj, kind))

    injected, consumed, dropped = 0, [], []

    # Group by host first so the per-host cap can be applied, and round-robin
    # across kinds within each host. Round-robin rather than first-come because
    # a class that always lost the cap would be the only one never rendered with
    # real displacement - a systematic per-class difference in appearance is a
    # cue a detector can learn instead of learning the defect.
    by_host: dict[str, dict[str, list]] = {}
    fatal_errors = []
    for instance_id, (plane, kind) in enumerate(planes, start=1):
        host = resolve_host(plane, state, host_by_name)
        if host is None:
            reason = "missing_host"
            logger.error("no host found for %s; dropping it", plane.name)
            _record_instance(
                plane, None, kind, instance_id, "none", "failed", reason=reason
            )
            dropped.append(plane)
            fatal_errors.append(f"{plane.name}:{reason}")
            continue
        if (
            not allow_nearest_host_fallback
            and plane.get("syndefect_host_resolution") != "solver_relation"
        ):
            reason = "unresolved_intended_host_relation"
            _record_instance(
                plane, host, kind, instance_id, "none", "failed", reason=reason
            )
            dropped.append(plane)
            fatal_errors.append(f"{plane.name}:{reason}")
            continue
        surface = plane.get("syndefect_surface")
        compatible, finish, substrate_note = substrate_compatibility(
            kind, surface, host
        )
        plane["syndefect_host_finish"] = finish
        if enforce_substrates and not compatible:
            _record_instance(
                plane,
                host,
                kind,
                instance_id,
                "none",
                "dropped",
                reason=substrate_note,
            )
            dropped.append(plane)
            continue

        support = None
        if validate_support:
            support = validate_defect_support(
                plane,
                host,
                distance_tolerance_m=support_distance_tolerance_m,
                normal_tolerance_deg=support_normal_tolerance_deg,
            )
            if not support["valid"]:
                reason = "invalid_support:" + ",".join(support["reasons"])
                _record_instance(
                    plane,
                    host,
                    kind,
                    instance_id,
                    "none",
                    "dropped",
                    reason=reason,
                    support=support,
                )
                dropped.append(plane)
                continue

        qa_views = _qa_camera_records(plane.name)
        if require_qa_view and not any(view["pose_valid"] for view in qa_views):
            reason = "missing_valid_qa_view"
            _record_instance(
                plane,
                host,
                kind,
                instance_id,
                "none",
                "dropped",
                reason=reason,
                support=support,
            )
            dropped.append(plane)
            continue
        by_host.setdefault(host.name, {}).setdefault(kind, []).append(
            (plane, host, instance_id, support)
        )

    for host_name in sorted(by_host):
        queues = {
            kind: sorted(items, key=lambda t: t[0].name)
            for kind, items in by_host[host_name].items()
        }
        kinds = sorted(queues)
        budget = max_per_host
        while any(queues[kind] for kind in kinds):
            for kind in kinds:
                if not queues[kind]:
                    continue
                plane, host, instance_id, support = queues[kind].pop(0)
                # The real limit is the host material's node count, not how many
                # defects share it. Measured: a material emits its AOVs at 3796
                # nodes and emits nothing at all at 4387 - the render still
                # succeeds and every defect on that host loses its labels
                # silently. Defect cost varies about tenfold between a crack and
                # a blister cluster, so a defect count cannot police this: six
                # cracks are safe where two blister clusters are not.
                over_node_budget = (
                    max_host_shader_nodes > 0
                    and _host_material_node_count(host) >= max_host_shader_nodes
                )
                if budget <= 0 or over_node_budget:
                    if drop_over_cap:
                        _record_instance(
                            plane,
                            host,
                            kind,
                            instance_id,
                            "none",
                            "dropped",
                            reason=(
                                "over_host_shader_node_budget"
                                if over_node_budget
                                else "over_host_shader_cap"
                            ),
                            support=support,
                        )
                        dropped.append(plane)
                    else:
                        _record_instance(
                            plane,
                            host,
                            kind,
                            instance_id,
                            "card",
                            "retained",
                            reason="over_host_shader_cap_card_fallback",
                            support=support,
                        )
                    continue
                if inject_defect(plane, host, kind, int_hash(plane.name), instance_id):
                    nodes_after = _host_material_node_count(host)
                    if 0 < HOST_SHADER_NODE_CLIFF <= nodes_after:
                        # Already over: this material will emit no AOVs, and the
                        # field cannot be taken back out. Fail loudly - a
                        # rejected scene is recoverable, a scene full of
                        # confidently mislabelled frames is not.
                        reason = (
                            f"host_shader_node_overflow:{nodes_after}"
                            f">={HOST_SHADER_NODE_CLIFF}"
                        )
                        logger.error(
                            "host %s reached %d shader nodes injecting %s; past the "
                            "measured AOV cliff, so every defect on this host would "
                            "lose its labels while the render still succeeded",
                            host.name,
                            nodes_after,
                            plane.name,
                        )
                        _record_instance(
                            plane, host, kind, instance_id, "none", "failed",
                            reason=reason, support=support,
                        )
                        dropped.append(plane)
                        fatal_errors.append(f"{plane.name}:{reason}")
                        continue
                    injected += 1
                    budget -= 1
                    consumed.append(plane)
                    _record_instance(
                        plane,
                        host,
                        kind,
                        instance_id,
                        "injected",
                        "retained",
                        support=support,
                    )
                else:
                    reason = str(
                        plane.get("syndefect_injection_error", "injection_failed")
                    )
                    _record_instance(
                        plane,
                        host,
                        kind,
                        instance_id,
                        "none",
                        "failed",
                        reason=reason,
                        support=support,
                    )
                    dropped.append(plane)
                    fatal_errors.append(f"{plane.name}:{reason}")

    if dropped:
        dropped_names = {plane.name for plane in dropped}
        if retarget_dropped_cameras:
            _retarget_orphaned_defect_cameras(dropped_names, consumed)
        else:
            _mark_orphaned_defect_cameras(dropped_names)
        logger.warning(
            "%d defect(s) dropped or failed; cards hidden: %s",
            len(dropped),
            ", ".join(sorted(p.name for p in dropped)[:8]),
        )

    if hide_planes:
        # Hidden, not deleted: the solver state still holds references to these
        # objects and serialising it after a delete raises ReferenceError on the
        # removed StructRNA. Hiding from render is enough - a hidden card
        # contributes to no pass, so it leaves the imagery and labels alone.
        #
        # preview_planes_in_viewport keeps the card in the viewport while still
        # excluding it from every render. The injected defect is a Cycles
        # displacement field, and neither Solid nor Material Preview shading can
        # show one - Material Preview is EEVEE, which has no adaptive
        # subdivision - so with the card hidden too there is nothing to look at
        # in the blend unless you switch the viewport to Cycles Rendered. The
        # card carries the same morphology from the same seed as real mesh, so
        # it is an honest proxy for eyeballing placement and scale.
        for plane in consumed:
            plane.hide_render = True
            plane.hide_viewport = not preview_planes_in_viewport
            plane["syndefect_injected"] = True
            plane["syndefect_viewport_proxy"] = bool(preview_planes_in_viewport)

    total = finalize_injections(dicing_rate=dicing_rate)
    logger.info(
        "defect injection: %d defect(s) into %d host(s); %d card(s) hidden",
        total,
        len(_REGISTRY),
        len(consumed) if hide_planes else 0,
    )
    if fatal_errors and fail_scene_on_injection_error:
        raise RuntimeError(
            "defect injection rejected the scene after explicit failure records: "
            + "; ".join(fatal_errors[:8])
        )
    return injected
