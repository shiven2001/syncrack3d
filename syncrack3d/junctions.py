"""Junction cracks: move a share of crack cards to structure edges (--crack-junction-fraction).

Real wall cracks concentrate where the wall meets a stiffer or framed element:
diagonal cracks from the corners of window, door, vent and hatch openings, and
cracks along the underside of sills, ledges and beams, beside door jambs and
along trims. Each selected card is moved to such an anchor and turned to run
away from the corner or along the edge; its size and seed are kept, so the crack
shape is the same draw. Cracks along an edge get a near-straight path through
the card's `cwg_crack_overrides`. Uses its own RNG stream: walls without
structures, and fraction 0, are unchanged.

Wall coordinates: u across the wall (0 at the centre), v up from the floor.
Crack direction is an angle in the wall plane, 0 = +u, 90 = +v, stored on the card
as `cwg_crack_angle_deg` (without it every crack runs along the wall's vertical).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace

import bpy

from syncrack3d.defects import _EDGE_MARGIN_M, _STANDOFF_M

OPENINGS = ("window", "door", "exterior_vent", "service_hatch", "vent", "access_panel")
SALT = 0x4A554E43


def _anchors(plan, r):
    """(name, (u, v), angle_deg, mode) candidates for one structure.

    mode "corner": the crack starts at the point and runs along the angle.
    mode "edge": the crack is centred on the point and runs along the angle.
    """
    k, x, z, w, h = plan.kind, plan.u_m, plan.v_m, plan.width_m, plan.height_m
    off = r.uniform(0.02, 0.05)
    out = []
    if k in OPENINGS:
        if k == "window":
            top, bot, half_w_top, half_w_bot = z + h / 2 + 0.12, z - h / 2 - 0.0675, w / 2 + 0.10, w / 2 + 0.08
        elif k == "door":
            top, bot, half_w_top, half_w_bot = z + h / 2, None, w / 2, w / 2
        else:
            top, bot, half_w_top, half_w_bot = z + h / 2 + 0.01, z - h / 2 - 0.01, w / 2 + 0.01, w / 2 + 0.01
        for sx in (-1, 1):
            a = r.uniform(30.0, 60.0)
            out.append((f"{k}_top_corner", (x + sx * half_w_top, top), a if sx > 0 else 180.0 - a, "corner"))
            if bot is not None:
                out.append((f"{k}_bottom_corner", (x + sx * half_w_bot, bot), -a if sx > 0 else 180.0 + a, "corner"))
        if k == "window":
            out.append(("sill_underside", (x + r.uniform(-0.35, 0.35) * w, bot - off), r.uniform(-4.0, 4.0), "edge"))
        if k == "door":
            for sx in (-1, 1):
                out.append(("door_jamb", (x + sx * (w / 2 + 0.03 + off), z + r.uniform(-0.2, 0.2) * h),
                            90.0 + r.uniform(-3.0, 3.0), "edge"))
    elif k == "ledge":
        out.append(("ledge_underside", (x + r.uniform(-0.35, 0.35) * w, z - h / 2 - 0.02 - off), r.uniform(-4.0, 4.0), "edge"))
        out.append(("ledge_top", (x + r.uniform(-0.35, 0.35) * w, z + h / 2 + off), r.uniform(-4.0, 4.0), "edge"))
    elif k == "ceiling_trim":
        out.append(("ceiling_trim", (x + r.uniform(-0.4, 0.4) * w, z - h / 2 - off), r.uniform(-3.0, 3.0), "edge"))
    elif k == "skirting":
        out.append(("skirting", (x + r.uniform(-0.4, 0.4) * w, z + h / 2 + off), r.uniform(-3.0, 3.0), "edge"))
    return out


def snap_to_junctions(placements, structure_plans, spec, seed, fraction):
    """Return (placements, records): a share of crack cards moved to structure anchors."""
    if fraction <= 0 or not structure_plans or not placements:
        return placements, []
    r = random.Random(int(seed) ^ SALT)
    anchors = [a for plan in structure_plans for a in _anchors(plan, r)]
    if not anchors:
        return placements, []
    half_wall = spec.width_m * 0.5 - _EDGE_MARGIN_M
    out, records = list(placements), []
    for i, p in enumerate(placements):
        if r.random() >= fraction:
            continue
        half = 0.42 * min(p.size_m, 0.42)  # crack half-length (defect_injection._inject_crack)
        for _ in range(12):
            name, (au, av), angle, mode = r.choice(anchors)
            d = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
            cu, cv = (au + d[0] * half * 0.95, av + d[1] * half * 0.95) if mode == "corner" else (au, av)
            reach = p.size_m * 0.5
            if not (-half_wall + reach <= cu <= half_wall - reach
                    and _EDGE_MARGIN_M + reach <= cv <= spec.height_m - _EDGE_MARGIN_M - reach):
                continue
            if any(j != i and math.hypot(cu - q.u_m, cv - q.v_m) < 0.4 * (p.size_m + q.size_m)
                   for j, q in enumerate(out)):
                continue
            card = bpy.data.objects[p.object_name]
            card.location = (cu, -_STANDOFF_M, cv)
            card["cwg_crack_angle_deg"] = float(angle)  # read by defect_injection.inject_defect
            if mode == "edge":
                card["cwg_crack_overrides"] = json.dumps({"crack_path_mode": "straight"})
            out[i] = replace(p, u_m=round(cu, 6), v_m=round(cv, 6), spin_deg=round(angle % 360.0, 3))
            records.append({"index": p.index, "anchor": name, "mode": mode, "u_m": round(cu, 4),
                            "v_m": round(cv, 4), "angle_deg": round(angle, 2)})
            break
    bpy.context.view_layer.update()
    return out, records
