# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei

import bpy
import numpy as np
from numpy.random import uniform

from cwg_infinigen.assets import colors
from cwg_infinigen.assets.materials.wood import wood
from cwg_infinigen.assets.utils.decorate import read_edge_center, read_edge_direction
from cwg_infinigen.assets.utils.mesh import bevel
from cwg_infinigen.assets.utils.object import new_cube
from cwg_infinigen.core.constraints.constraint_language.constants import RoomConstants
from cwg_infinigen.core.placement.factory import AssetFactory
from cwg_infinigen.core.util import blender as butil
from cwg_infinigen.core.util.math import FixedSeed


class DoorCasingFactory(AssetFactory):
    # Gap kept between the back of the architrave moulding and the wall plane.
    wall_clearance = 0.010

    def __init__(self, factory_seed, coarse=False, constants=None):
        super(DoorCasingFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            if constants is None:
                constants = RoomConstants()
            self.constants = constants
            # ~90-150 mm of architrave.
            self.margin = constants.door_size * uniform(0.04, 0.065)
            # Architrave stands proud of the skirting (20-50 mm) so the board
            # dies into it instead of poking past its face.
            self.extrude = uniform(0.032, 0.052)
            # Depth the moulding profile is swept back from that proud face.
            # Never the full `extrude`: the casing spans +-(wall_thickness/2 +
            # extrude) about the door plane, so a profile swept the whole depth
            # ends exactly on the wall surface at +-wall_thickness/2. The STEPS
            # preset finishes in a flat ring, which then sits 0.0 mm from the
            # wall over ~0.08 m2 - two surfaces at one depth, so the wall wins
            # part of the ring and paints a band of wall texture across the
            # frame. `bevel` draws its preset from five uniformly, so that was
            # about one door in five. CORNICE and CROWN cleared the wall by only
            # 9-13 mm, which is inside the relief walls carry anyway (plaster
            # bump, and in defect scenes a displaced blister field), so the
            # clearance is subtracted for every preset rather than special-cased.
            self.profile_depth = max(
                self.extrude - self.wall_clearance, self.extrude * 0.4
            )
            self.bevel_all_sides = uniform() < 0.3
            self.surface = wood.InteriorWood()
            self.metal_color = colors.metal_hsv()

    def create_asset(self, **params) -> bpy.types.Object:
        obj = new_cube()
        obj.location = 0, 0, 1
        butil.apply_transform(obj, True)
        w = self.constants.door_width
        s = self.constants.door_size
        obj.scale = (
            w / 2 + self.margin,
            self.constants.wall_thickness / 2 + self.extrude,
            s / 2 + self.margin / 2,
        )
        butil.apply_transform(obj)
        cutter = new_cube()
        cutter.location = 0, 0, 1 - 1e-3
        butil.apply_transform(cutter, True)
        cutter.scale = w / 2 - 1e-3, self.constants.wall_thickness + self.extrude, s / 2
        butil.apply_transform(cutter)
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
        butil.delete(cutter)

        x, y, z = read_edge_center(obj).T
        x_, y_, z_ = read_edge_direction(obj).T

        if self.bevel_all_sides:
            selection = (np.abs(z_) > 0.5) | (np.abs(x_) > 0.5)
        else:
            selection = ((np.abs(z_) > 0.5) & (np.abs(x) < w / 2 + self.margin / 2)) | (
                (np.abs(x_) > 0.5) & (z < s + self.margin / 2)
            )
        # Blender 4.x keeps bevel weight in a named edge attribute rather than on
        # the edge struct, so the modifier reads nothing unless we create it.
        attr = obj.data.attributes.get("bevel_weight_edge")
        if attr is None:
            attr = obj.data.attributes.new("bevel_weight_edge", "FLOAT", "EDGE")
        attr.data.foreach_set("value", selection.astype(float))
        bevel(obj, self.profile_depth, limit_method="WEIGHT")
        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets, metal_color=self.metal_color)
