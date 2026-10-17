# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Per-class defect coverage passes, written as Cycles AOVs.

The material-index pass labels a pixel by whichever material the camera ray
happened to hit, so it is a hard, aliased, one-sample-per-pixel decision, and it
can only ever describe a defect that owns its own material. Neither property
survives moving a defect into its host wall's shader, where the crack is a field
inside the wall material rather than a separate object.

An AOV sidesteps both. Each defect material writes its own coverage signal - the
same socket that drives Principled ``Alpha``, so the pass agrees with what is
actually opaque - into a VALUE pass named for its class. Cycles averages that
over all samples, so the result is an antialiased coverage map in 0..1 rather
than a jagged binary one, and it stays correct when the defect is invisible in
RGB because the geometry, not the shading, decides it.
"""

import logging

import bpy
import gin

logger = logging.getLogger(__name__)

# Prefix -> (class id, AOV name). Class ids must stay in step with
# DEFECT_PREFIXES in tools/prepare_defect_annotated_dataset.py. Longer prefixes
# are listed first so SpallingPlugMaterial does not match SpallingMaterial.
DEFECT_AOV_CLASSES = {
    "CrackMaterial": (1, "DefectCrack"),
    "PaintPeelMaterial": (2, "DefectPaintPeel"),
    "SpallingPlugMaterial": (3, "DefectSpalling"),
    "SpallingMaterial": (3, "DefectSpalling"),
    "BubbleMaterial": (4, "DefectPaintBubble"),
    "OpenWiringMaterial": (5, "DefectExposedWiring"),
    "PaintRunMaterial": (6, "DefectPaintRun"),
    "PaintPatchMaterial": (7, "DefectPaintPatch"),
    "CornerChipMaterial": (8, "DefectCornerChip"),
    "TileChipMaterial": (9, "DefectTileChip"),
}

AOV_NAME_TO_CLASS = {aov: class_id for _, (class_id, aov) in DEFECT_AOV_CLASSES.items()}

# These three AOVs form the instance-identity contract.  Cycles averages AOVs
# over pixel samples, so one scalar id is not enough: id 4 at 50% coverage is
# indistinguishable from id 2 at 100% coverage.  Dividing the first and second
# moments by coverage recovers a stable id for a single owner and exposes mixed
# owners as non-zero variance.  The exporter marks those mixed pixels ambiguous
# instead of silently assigning the wrong object.
DEFECT_INSTANCE_AOV = "DefectInstance"
DEFECT_INSTANCE_COVERAGE_AOV = "DefectInstanceCoverage"
DEFECT_INSTANCE_SQUARED_AOV = "DefectInstanceSquared"
DEFECT_INSTANCE_AOVS = (
    DEFECT_INSTANCE_AOV,
    DEFECT_INSTANCE_COVERAGE_AOV,
    DEFECT_INSTANCE_SQUARED_AOV,
)


def scene_has_instance_aov() -> bool:
    """True if any material writes the instance AOV, i.e. injection ran."""
    for material in bpy.data.materials:
        tree = material.node_tree
        if tree is None:
            continue
        for node in tree.nodes:
            if (
                node.bl_idname == "ShaderNodeOutputAOV"
                and node.aov_name == DEFECT_INSTANCE_AOV
            ):
                return True
    return False


def classify_material(material: bpy.types.Material) -> tuple[int, str] | None:
    """(class id, AOV name) for a defect material, or None if it is not one."""
    name = material.name or ""
    for prefix, entry in DEFECT_AOV_CLASSES.items():
        if name.startswith(prefix):
            return entry
    return None


def _coverage_socket(material: bpy.types.Material):
    """The socket describing how much of the pixel this defect actually covers.

    Defect factories drive Principled ``Alpha`` from their own mask, so that
    socket is the coverage. Where it is an unlinked constant the material is
    opaque wherever it is assigned, and a constant 1.0 is the right answer.
    """
    tree = material.node_tree
    if tree is None:
        return None
    for node in tree.nodes:
        if node.bl_idname != "ShaderNodeBsdfPrincipled":
            continue
        alpha = node.inputs.get("Alpha")
        if alpha is not None and alpha.is_linked:
            return alpha.links[0].from_socket
    # Fall back to a Mix Shader against a transparent BSDF, which some
    # factories use instead of the Alpha input.
    for node in tree.nodes:
        if node.bl_idname != "ShaderNodeMixShader":
            continue
        for link in node.inputs[1].links:
            if link.from_node.bl_idname == "ShaderNodeBsdfTransparent":
                fac = node.inputs["Fac"]
                if fac.is_linked:
                    return fac.links[0].from_socket
    return None


def wire_material_aov(material: bpy.types.Material, aov_name: str) -> bool:
    """Add an AOV output writing this material's coverage. Idempotent."""
    tree = material.node_tree
    if tree is None:
        return False
    for node in tree.nodes:
        if node.bl_idname == "ShaderNodeOutputAOV" and node.aov_name == aov_name:
            return True
    node = tree.nodes.new("ShaderNodeOutputAOV")
    node.aov_name = aov_name
    node.label = f"defect AOV: {aov_name}"
    coverage = _coverage_socket(material)
    if coverage is not None:
        tree.links.new(coverage, node.inputs["Value"])
    else:
        node.inputs["Value"].default_value = 1.0
    return True


def _ensure_view_layer_aov(view_layer, aov_name: str) -> None:
    for aov in view_layer.aovs:
        if aov.name == aov_name:
            aov.type = "VALUE"
            return
    aov = view_layer.aovs.add()
    aov.name = aov_name
    aov.type = "VALUE"


@gin.configurable
def setup_defect_aovs(view_layer=None, enabled: bool = False) -> list[str]:
    """Register a VALUE AOV per defect class present, and wire the materials.

    Returns the AOV names in a stable order so the compositor and the exporter
    agree on them. Off by default: turning it on adds a render pass per class.
    """
    if not enabled:
        return []
    if view_layer is None:
        view_layer = bpy.context.scene.view_layers["ViewLayer"]

    by_aov: dict[str, list[bpy.types.Material]] = {}
    for material in bpy.data.materials:
        entry = classify_material(material)
        if entry is None:
            continue
        by_aov.setdefault(entry[1], []).append(material)

    for aov_name in sorted(by_aov):
        _ensure_view_layer_aov(view_layer, aov_name)
        for material in by_aov[aov_name]:
            wire_material_aov(material, aov_name)
        logger.info(
            "defect AOV %s covers %d material(s)", aov_name, len(by_aov[aov_name])
        )
    names = sorted(by_aov)
    # Injection writes the instance AOVs into host materials, whose names are the
    # host's, not a defect prefix, so it is not discovered by the sweep above.
    if scene_has_instance_aov():
        for aov_name in DEFECT_INSTANCE_AOVS:
            _ensure_view_layer_aov(view_layer, aov_name)
            names.append(aov_name)
        logger.info("defect instance AOVs registered: %s", DEFECT_INSTANCE_AOVS)
    if not by_aov:
        logger.info("no defect coverage materials found")
    return names
