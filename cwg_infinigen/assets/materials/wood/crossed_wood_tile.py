# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.


# Authors: Lingjie Mei
from cwg_infinigen.assets.materials.ceramic import tile

from .wood import shader_wood_floor


# The shared tile shader recesses joints 16-32 mm and adds several millimetres
# of noise. On staggered and square boards that reads as plank thickness. On
# crossed parquet the two grain directions stack and the measured relief was
# about 25 mm. Scaling every displacement node by 0.08 brings that to about 2 mm
# without drawing any new random numbers, so a crack on the same wall is unchanged.
_PARQUET_RELIEF_SCALE = 0.08


def _scale_displacement(material, factor: float) -> None:
    for node in material.node_tree.nodes:
        if node.bl_idname != "ShaderNodeDisplacement":
            continue
        scale = node.inputs.get("Scale")
        if scale is not None and not scale.is_linked:
            scale.default_value *= factor


class CrossedWoodTile:
    def generate(self):
        shader_func = shader_wood_floor
        material = tile.Tile().generate(shader_func=shader_func, shape="crossed")
        _scale_displacement(material, _PARQUET_RELIEF_SCALE)
        return material

    __call__ = generate
    # def make_sphere():e
