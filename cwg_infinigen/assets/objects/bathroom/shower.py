"""Corner walk-in shower: acrylic tray, chrome frame, L-shaped glass, rain head.

Tray 950–1200 x 1050–1400 mm, glass 1950–2050 mm.
Mixer ~1050–1150 mm AFF, rain head ~2050–2150 mm AFF, walk-in ~45–58% of the run.
Placeholder matches BathtubFactory: back wall at min-X.
"""

import numpy as np
from numpy.random import uniform

from cwg_infinigen.assets.objects.wall_decorations.primitives import (
    assign,
    box,
    cylinder,
    glass_material,
    shade_smooth,
    solid_material,
)
from cwg_infinigen.assets.utils.object import join_objects, new_bbox
from cwg_infinigen.core.placement.factory import AssetFactory
from cwg_infinigen.core.util import blender as butil
from cwg_infinigen.assets.objects.bathroom.fittings import bathroom_chrome
from cwg_infinigen.core.util.math import FixedSeed


# Highest point of the stall, glass or rain head.
#
# A corner cubicle sits under the perimeter downstand beam, whose soffit is at
# `wall_height - uniform(0.24, 0.32)` - 2.48 m above the floor at worst by
# construction, and 2.575 m measured on a generated flat. Holding the stall at
# 2.15 m leaves a third of a metre, so the cubicle can grow in plan without ever
# meeting the beam. Grow the plan freely; raise this only against that number.
MAX_STALL_TOP = 2.15


class ShowerStallFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            # Plan size. The old 0.80-0.92 x 0.82-0.96 is a small-flat cubicle,
            # and it is the minimum a body fits in - but these bathrooms solve
            # at 14-16 m2, where an 800 mm tray reads as a shower for a doll's
            # house. Stock trays run 800/900/1000 mm square and 1200x900 in
            # rectangle, so this range covers the middle and upper of what is
            # actually sold and suits the room it lands in.
            self.size = uniform(0.95, 1.20)  # into room (+X from back wall)
            self.width = uniform(1.05, 1.40)  # along wall (+Y)
            # Screens are sold at 1900-2000 mm; 1850 was short.
            self.glass_h = uniform(1.95, 2.05)
            self.tray_h = uniform(0.050, 0.070)
            self.rim = uniform(0.032, 0.042)
            self.glass_t = uniform(0.006, 0.008)
            self.post = uniform(0.018, 0.024)
            # Walk-in gap as a share of the run, not a fixed 500-600 mm that
            # would leave a wide tray almost entirely screened.
            self.opening = self.width * uniform(0.45, 0.58)
            # 220-300 mm rain heads.
            self.head_r = uniform(0.110, 0.150)
            # Arm scales with the tray so the head stays over its centre; a
            # fixed 300-400 mm left the head near the back wall once the stall
            # got deeper.
            self.arm_len = self.size * uniform(0.42, 0.52)
            self.head_z = uniform(2.05, MAX_STALL_TOP - self.FLANGE_R - 0.006)
            self.mixer_z = uniform(1.05, 1.15)
            assert self.top_z <= MAX_STALL_TOP

    # The door handle stands this far proud of the glass line, and the
    # rain-head flange - a disc turned on its side against the back wall -
    # reaches this far above the arm. The solver reserves exactly the
    # placeholder box, so both belong in it: without them the stall occupied
    # 26 mm of depth and 24 mm of height the solver believed were free. Read
    # off the geometry in create_asset, not guessed.
    HANDLE_PROUD = 0.026
    FLANGE_R = 0.024

    @property
    def top_z(self):
        return max(self.glass_h, self.head_z + self.FLANGE_R)

    def create_placeholder(self, **params):
        return new_bbox(
            -self.size, self.HANDLE_PROUD, 0, self.width, 0, self.top_z
        )

    def create_asset(self, **params):
        chrome = bathroom_chrome()
        acrylic = solid_material(
            f"ShowerTray_{self.factory_seed}",
            (0.93, 0.935, 0.94),
            roughness=0.22,
        )
        basin = solid_material(
            f"ShowerBasin_{self.factory_seed}",
            (0.86, 0.87, 0.88),
            roughness=0.28,
        )
        glass = glass_material(f"ShowerGlass_{self.factory_seed}")
        face_mat = solid_material(
            f"ShowerFace_{self.factory_seed}",
            (0.50, 0.51, 0.53),
            roughness=0.38,
            metallic=0.65,
        )
        parts = []

        d, w, th = self.size, self.width, self.tray_h
        x_wall = -d
        x_front = 0.0

        tray = box((d, w, th), location=(-d / 2, w / 2, th / 2), name="shower_tray")
        assign(tray, acrylic)
        butil.modify_mesh(tray, "BEVEL", width=0.006, segments=2)
        parts.append(tray)

        inner_d = d - self.rim * 2
        inner_w = w - self.rim * 2
        well = box(
            (inner_d, inner_w, 0.012),
            location=(-d / 2, w / 2, th - 0.004),
            name="shower_well",
        )
        assign(well, basin)
        parts.append(well)

        waste = cylinder(
            0.021,
            0.006,
            location=(-d / 2, w / 2, th + 0.001),
            name="shower_waste",
        )
        assign(waste, chrome)
        shade_smooth(waste)
        parts.append(waste)
        waste_hole = cylinder(
            0.012,
            0.004,
            location=(-d / 2, w / 2, th + 0.004),
            name="shower_waste_hole",
        )
        assign(
            waste_hole,
            solid_material(
                f"ShowerWasteHole_{self.factory_seed}", (0.06, 0.06, 0.06), roughness=0.7
            ),
        )
        parts.append(waste_hole)

        gz0 = th + 0.006
        gh = self.glass_h - gz0
        gz = gz0 + gh / 2
        p = self.post
        g_t = self.glass_t
        open_y = self.opening

        # Front screen at the room-facing side, walk-in gap near y=0.
        g_front_w = max(w - open_y - p, 0.22)
        g_front = box(
            (g_t, g_front_w, gh),
            location=(x_front - g_t / 2, w - p - g_front_w / 2, gz),
            name="shower_glass_front",
        )
        assign(g_front, glass)
        parts.append(g_front)

        # Return panel along the far side wall of the cubicle.
        g_side = box(
            (d - p * 1.2, g_t, gh),
            location=(-d / 2 + p * 0.2, w - g_t / 2, gz),
            name="shower_glass_side",
        )
        assign(g_side, glass)
        parts.append(g_side)

        def post_at(x, y):
            obj = box((p, p, self.glass_h), location=(x, y, self.glass_h / 2), name="shower_post")
            assign(obj, chrome)
            butil.modify_mesh(obj, "BEVEL", width=0.002, segments=2)
            shade_smooth(obj)
            return obj

        parts.append(post_at(x_front - p / 2, w - p / 2))
        parts.append(post_at(x_front - p / 2, open_y + p / 2))
        parts.append(post_at(x_wall + p / 2, w - p / 2))

        rail_f = box(
            (p * 0.65, g_front_w + p, p * 0.65),
            location=(x_front - p / 2, w - p - g_front_w / 2, self.glass_h - p * 0.3),
            name="shower_rail_front",
        )
        rail_s = box(
            (d - p * 0.4, p * 0.65, p * 0.65),
            location=(-d / 2 + p * 0.1, w - p / 2, self.glass_h - p * 0.3),
            name="shower_rail_side",
        )
        assign(rail_f, chrome)
        assign(rail_s, chrome)
        parts.extend([rail_f, rail_s])

        handle = cylinder(
            0.008,
            0.22,
            location=(x_front + 0.018, open_y + p + 0.06, 1.05),
            name="shower_handle",
        )
        assign(handle, chrome)
        shade_smooth(handle)
        parts.append(handle)
        for hz in (0.94, 1.16):
            stub = cylinder(
                0.007,
                0.028,
                location=(x_front + 0.004, open_y + p + 0.06, hz),
                rotation=(0, np.pi / 2, 0),
                name="shower_handle_stub",
            )
            assign(stub, chrome)
            shade_smooth(stub)
            parts.append(stub)

        # Mixer on the tiled back wall (~1100 mm AFF). Back face of the plate
        # sits on the placeholder's min-X so flush_fixture leaves no gap.
        plate_t = 0.012
        mix_x = x_wall + plate_t / 2
        mix_y = w * 0.42
        mix_plate = cylinder(
            0.055,
            plate_t,
            location=(mix_x, mix_y, self.mixer_z),
            rotation=(0, np.pi / 2, 0),
            name="shower_mixer_plate",
        )
        assign(mix_plate, chrome)
        shade_smooth(mix_plate)
        parts.append(mix_plate)
        mix_body = cylinder(
            0.028,
            0.022,
            location=(mix_x + 0.016, mix_y, self.mixer_z),
            rotation=(0, np.pi / 2, 0),
            name="shower_mixer_body",
        )
        assign(mix_body, chrome)
        shade_smooth(mix_body)
        parts.append(mix_body)
        lever = box(
            (0.012, 0.014, 0.09),
            location=(mix_x + 0.034, mix_y, self.mixer_z + 0.04),
            name="shower_lever",
        )
        assign(lever, chrome)
        butil.modify_mesh(lever, "BEVEL", width=0.002, segments=2)
        parts.append(lever)

        # Wall arm + rain head (~2050 mm AFF), projecting into the stall.
        arm_y = w * 0.50
        arm = cylinder(
            0.010,
            self.arm_len,
            location=(x_wall + 0.02 + self.arm_len / 2, arm_y, self.head_z),
            rotation=(0, np.pi / 2, 0),
            name="shower_arm",
        )
        assign(arm, chrome)
        shade_smooth(arm)
        parts.append(arm)
        flange = cylinder(
            0.024,
            plate_t,
            location=(x_wall + plate_t / 2, arm_y, self.head_z),
            rotation=(0, np.pi / 2, 0),
            name="shower_flange",
        )
        assign(flange, chrome)
        shade_smooth(flange)
        parts.append(flange)
        head_x = x_wall + 0.02 + self.arm_len
        head = cylinder(
            self.head_r,
            0.016,
            location=(head_x, arm_y, self.head_z - 0.008),
            name="shower_head",
        )
        assign(head, chrome)
        shade_smooth(head)
        parts.append(head)
        face = cylinder(
            self.head_r * 0.90,
            0.004,
            location=(head_x, arm_y, self.head_z - 0.018),
            name="shower_face",
        )
        assign(face, face_mat)
        parts.append(face)
        for i in range(8):
            ang = i * (2 * np.pi / 8)
            rr = self.head_r * 0.55
            hole = cylinder(
                0.004,
                0.005,
                location=(
                    head_x,
                    arm_y + rr * np.cos(ang),
                    self.head_z - 0.021,
                ),
                name="shower_nozzle",
            )
            assign(hole, face_mat)
            parts.append(hole)

        # Handheld on a slide rail — typical apartment fit-out.
        cap_t = 0.012
        cap_x = x_wall + cap_t / 2
        rail_x = x_wall + 0.018
        rail_y = w * 0.18
        slider = cylinder(
            0.009,
            0.72,
            location=(rail_x, rail_y, 1.28),
            name="shower_slider",
        )
        assign(slider, chrome)
        shade_smooth(slider)
        parts.append(slider)
        for zz in (0.92, 1.64):
            cap = cylinder(
                0.016,
                cap_t,
                location=(cap_x, rail_y, zz),
                rotation=(0, np.pi / 2, 0),
                name="shower_slider_cap",
            )
            assign(cap, chrome)
            shade_smooth(cap)
            parts.append(cap)
        hand = cylinder(
            0.022,
            0.085,
            location=(rail_x + 0.05, rail_y, 1.22),
            rotation=(0.45, np.pi / 2, 0),
            name="shower_handheld",
        )
        assign(hand, chrome)
        shade_smooth(hand)
        parts.append(hand)

        obj = join_objects(parts)
        obj.name = f"ShowerStallFactory({self.factory_seed}).stall"
        return obj
