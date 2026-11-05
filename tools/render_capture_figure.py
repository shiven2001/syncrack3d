"""The paper's camera and light sampling figure, one column wide (\\columnwidth, 600 dpi).

Every point is a draw from the generator's own samplers, syncrack3d.cameras and
syncrack3d.lighting, with the unified recipe's settings. The walls and crack tokens
are the ones logged by tools/render_placement_figure.py trace. No point is placed by
hand, and the one rig that carries the symbols is picked from the draws by a rule.

The draws use their own random stream, so they follow the generator's distributions
but are not the rigs of those walls' released images. Views that build_dataset drops
because a fixture blocks the lens are not modelled, and neither are fixture views
(the unified recipe's --structure-views).

    # draws -> outputs/capture_figure/draws.json, prints a check against the paper's tables
    PYTHONPATH=. .venv/bin/python tools/render_capture_figure.py sample

    # -> outputs/capture_figure/capture_sampling.png
    .venv/bin/python tools/render_capture_figure.py plot

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "outputs" / "capture_figure"
TRACES = REPO / "outputs" / "placement_figure" / "traces"
DRAWS = OUT / "draws.json"
FIGURE = OUT / "capture_sampling.png"

COLUMN_WIDTH_IN = 3.297  # \columnwidth of cas-dc: 238.25 pt
FIGURE_DPI = 600
STREAM = 0x43415054

# The unified recipe's capture flags (syncrack3d/recipes.py: COMMON) and build_dataset's
# defaults for the rest. Limits are repeated in plot() only to draw them.
CAPTURE = dict(closeups_per_defect=2, context_view_fraction=0.25, frame_fill_range=(0.6, 1.5),
               max_yaw_deg=38.0, max_pitch_deg=26.0, preserve_closeup_scale=True, aim_jitter=0.3)
D_NEAR_M = 0.22
PLOTTED_CLOSEUPS, PLOTTED_CONTEXT, PLOTTED_LIGHTS = 300, 91, 320  # the sampled share of each role


def sample() -> None:
    import bpy

    from syncrack3d.cameras import sample_camera_poses
    from syncrack3d.lighting import sample_lighting_plan

    traces = sorted(TRACES.glob("seed_*.json"))
    if not traces:
        raise SystemExit(f"no traces in {TRACES}: run tools/render_placement_figure.py trace first")
    collection = bpy.data.collections.new("capture_figure")
    bpy.context.scene.collection.children.link(collection)

    cameras, lights = [], []
    for path in traces:
        trace = json.loads(path.read_text())
        rng = random.Random(trace["seed"] ^ STREAM)
        spec = SimpleNamespace(width_m=trace["width_m"], height_m=trace["height_m"])
        placements = [SimpleNamespace(index=t["index"], u_m=t["final"][0], v_m=t["final"][1], size_m=t["size_m"],
                                      object_name=f"token_{t['index']}") for t in trace["tokens"]]
        lighting = sample_lighting_plan(rng, spec.width_m, spec.height_m)
        key = lighting.lights[0]
        centre = (0.0, 0.0, spec.height_m * 0.5)
        lights.append(dict(regime=key.regime, incidence_deg=key.incidence_deg, azimuth_deg=key.azimuth_deg,
                           distance_m=math.dist(key.location, centre), kelvin=key.kelvin,
                           width_m=spec.width_m, height_m=spec.height_m,
                           has_fill=len(lighting.lights) > 1))
        for pose in sample_camera_poses(rng, spec, placements, collection, **CAPTURE):
            cameras.append(dict(role=pose.role, distance_m=pose.distance_m, yaw_deg=pose.yaw_deg,
                                pitch_deg=pose.pitch_deg, lens_equiv_mm=pose.lens_equiv_mm))
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)

    OUT.mkdir(parents=True, exist_ok=True)
    DRAWS.write_text(json.dumps(dict(walls=len(traces), cameras=cameras, lights=lights)) + "\n")
    check()


def check() -> None:
    """The draws against the bands and chances stated in the paper's tables."""
    data = json.loads(DRAWS.read_text())
    cameras, lights = data["cameras"], data["lights"]
    print(f"walls {data['walls']}, cameras {len(cameras)}, key lights {len(lights)}")
    for role in ("closeup", "context"):
        chosen = [c for c in cameras if c["role"] == role]
        d = sorted(c["distance_m"] for c in chosen)
        print(f"{role}: n={len(chosen)} d={d[0]:.2f}-{d[-1]:.2f} median={d[len(d) // 2]:.2f} "
              f"at d_near={sum(x <= D_NEAR_M + 1e-3 for x in d) / len(d):.2f} "
              f"yaw={min(c['yaw_deg'] for c in chosen):.1f}..{max(c['yaw_deg'] for c in chosen):.1f} "
              f"pitch={min(c['pitch_deg'] for c in chosen):.1f}..{max(c['pitch_deg'] for c in chosen):.1f}")
    for regime in sorted({light["regime"] for light in lights}):
        chosen = [light for light in lights if light["regime"] == regime]
        print(f"{regime}: share={len(chosen) / len(lights):.2f} "
              f"incidence={min(c['incidence_deg'] for c in chosen):.1f}-{max(c['incidence_deg'] for c in chosen):.1f} "
              f"K={min(c['kelvin'] for c in chosen):.0f}-{max(c['kelvin'] for c in chosen):.0f}")
    print(f"fill share={sum(light['has_fill'] for light in lights) / len(lights):.2f}")
    # A sampled source can stand beyond the wall's own span (the sampler does not clip it to the room).
    def along(light):
        alpha, beta = math.radians(light["incidence_deg"]), math.radians(light["azimuth_deg"])
        return (light["distance_m"] * math.sin(alpha) * math.cos(beta),
                light["distance_m"] * math.sin(alpha) * math.sin(beta))
    over = [abs(along(light)[1]) > light["height_m"] / 2 for light in lights]
    wide = [abs(along(light)[0]) > light["width_m"] / 2 for light in lights]
    print(f"key lights past floor or ceiling height={sum(over) / len(lights):.2f} "
          f"past wall ends={sum(wide) / len(lights):.2f} either={sum(a or b for a, b in zip(over, wide)) / len(lights):.2f}")
    for regime in sorted({light["regime"] for light in lights}):
        chosen = [o for o, light in zip(over, lights) if light["regime"] == regime]
        print(f"  {regime}: past floor or ceiling={sum(chosen) / len(chosen):.2f}")


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import patheffects
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Arc

    # Roles stay neutral: in this figure colour means one thing, the light regime.
    CLOSEUP, CONTEXT, INK, MUTED, WALL = "#222222", "#A3A3A3", "#1A1A1A", "#5F5F5F", "#4A4A4A"
    FONT_PT = 6.5
    # Colour and marker both carry the regime, so it survives greyscale printing.
    REGIMES = (("window_raking", "Raking window", "#0072B2", "o"),
               ("window_soft", "Soft window", "#D55E00", "s"),
               ("ceiling_diffuse", "Diffuse ceiling", "#009E73", "^"),
               ("lamp_mixed", "Mixed lamp", "#CC79A7", "D"),
               ("flash", "Flash", "#E69F00", "v"))
    data = json.loads(DRAWS.read_text())
    rng = random.Random(STREAM)
    closeups = rng.sample([c for c in data["cameras"] if c["role"] == "closeup"], PLOTTED_CLOSEUPS)
    context = rng.sample([c for c in data["cameras"] if c["role"] == "context"], PLOTTED_CONTEXT)
    lights = rng.sample(data["lights"], PLOTTED_LIGHTS)
    halo = [patheffects.withStroke(linewidth=1.4, foreground="white")]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_PT, "mathtext.fontset": "dejavusans",
                         "axes.titlesize": FONT_PT, "axes.linewidth": 0.5})

    def plan(camera):
        """Eq. (camera_pose) seen from above: along the wall and out from it. The angle from the normal is yaw."""
        yaw, pitch = math.radians(camera["yaw_deg"]), math.radians(camera["pitch_deg"])
        reach = camera["distance_m"] * math.cos(pitch)
        return reach * math.sin(yaw), reach * math.cos(yaw)

    def elevation(camera):
        """The same pose in the vertical plane through camera and aim point: the angle is pitch, the length d."""
        pitch = math.radians(camera["pitch_deg"])
        return camera["distance_m"] * math.cos(pitch), camera["distance_m"] * math.sin(pitch)

    def style(ax):
        ax.set_aspect("equal")
        ax.tick_params(length=2, width=0.5, pad=1.5, labelsize=FONT_PT - 0.5, color=MUTED)
        for name in ("top", "right"):
            ax.spines[name].set_visible(False)
        for name in ("left", "bottom"):
            ax.spines[name].set_color(MUTED)

    # Both camera panels share a scale, so distances compare between them.
    fig_w, left, gap, right = COLUMN_WIDTH_IN, 0.27, 0.35, 0.05
    plan_x, depth, side_z = (-2.2, 2.2), 3.4, (-1.75, 1.65)
    unit = (fig_w - left - gap - right) / (plan_x[1] - plan_x[0] + depth)
    plan_w, cam_w, cam_h = unit * (plan_x[1] - plan_x[0]), unit * depth, unit * depth
    light_h, top, x_labels, legend_h, title_h, legend2_h = cam_h, 0.13, 0.27, 0.15, 0.16, 0.25
    y_light = legend2_h + x_labels
    y_legend = y_light + light_h + title_h
    y_cam = y_legend + legend_h + x_labels
    fig_h = y_cam + cam_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))

    def axes(x_in, y_in, w_in, h_in, **kw):
        return fig.add_axes((x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h), **kw)

    # (a) A true plan: the wall along the top, the room below it.
    ax = axes(left, y_cam, plan_w, cam_h)
    style(ax)
    ax.set_xlim(*plan_x)
    ax.set_ylim(depth, 0)
    ax.set_title("(a) Cameras, plan view", pad=3)
    ax.set_xlabel("Along the wall (m)", labelpad=1.5)
    ax.set_ylabel("Out from the wall (m)", labelpad=1.5)
    ax.plot(plan_x, (0, 0), color=WALL, lw=1.6, solid_capstyle="butt", clip_on=False, zorder=2)
    ax.text(plan_x[1], 0.07, "Wall", ha="right", va="top", color=INK, zorder=8)
    limit = math.radians(CAPTURE["max_yaw_deg"])
    for sign in (-1, 1):
        ax.plot([0, sign * depth * math.tan(limit)], [0, depth], color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.plot((0, 0), (0, depth), color=MUTED, lw=0.4, zorder=1)
    ax.text(plan_x[1], plan_x[1] / math.tan(limit) + 0.1, f"±{CAPTURE['max_yaw_deg']:.0f}°", ha="right",
            va="top", color=INK, path_effects=halo, zorder=8)
    ax.annotate(r"$\mathbf{a}$", xy=(0, 0.015), xytext=(-1.25, 0.42), ha="center", va="center", color=INK, zorder=8,
                arrowprops=dict(arrowstyle="-|>,head_width=0.09,head_length=0.16", color=INK, lw=0.5, shrinkA=1,
                                shrinkB=0))

    # (b) The vertical plane through the camera, floor below.
    ax_side = axes(left + plan_w + gap, y_cam, cam_w, cam_h)
    style(ax_side)
    ax_side.set_xlim(0, depth)
    ax_side.set_ylim(*side_z)
    ax_side.set_title("(b) Cameras, side view", pad=3)
    ax_side.set_xlabel("Horizontal from $\\mathbf{a}$ (m)", labelpad=1.5)
    ax_side.set_ylabel("Height from $\\mathbf{a}$ (m)", labelpad=1.5)
    ax_side.plot((0, 0), side_z, color=WALL, lw=1.6, solid_capstyle="butt", clip_on=False, zorder=2)
    ax_side.plot((0, depth), (0, 0), color=MUTED, lw=0.4, zorder=1)
    ax_side.text(0.07, side_z[1], "Wall", ha="left", va="top", color=INK, zorder=8)

    for chosen, colour, size in ((context, CONTEXT, 4.0), (closeups, CLOSEUP, 1.8)):
        ax.scatter(*zip(*[plan(c) for c in chosen]), s=size, color=colour, lw=0, alpha=0.8, zorder=3)
        ax_side.scatter(*zip(*[elevation(c) for c in chosen]), s=size, color=colour, lw=0, alpha=0.8, zorder=3)

    # The rig that carries the symbols: the context draw nearest a pose where the angles read clearly.
    example = min(context, key=lambda c: abs(c["yaw_deg"] - 24) + abs(c["pitch_deg"] - 12) + 8 * abs(c["distance_m"] - 2.6))
    arc = dict(color=INK, lw=0.5, zorder=6)
    ex, ey = plan(example)
    yaw_rad = math.radians(example["yaw_deg"])
    ax.plot([0, ex], [0, ey], color=INK, lw=0.7, zorder=5)
    ax.plot(ex, ey, "o", ms=2.8, mfc=CONTEXT, mec=INK, mew=0.5, zorder=7)
    # Axes run downward here, so the arc is mirrored: theta is measured clockwise from +x on screen.
    ax.add_patch(Arc((0, 0), 2.0, 2.0, theta1=90 - example["yaw_deg"], theta2=90, **arc))
    ax.text(1.15 * math.sin(yaw_rad / 2), 1.15 * math.cos(yaw_rad / 2), r"$\psi$", ha="center", va="center",
            path_effects=halo, zorder=8)
    ax.text(ex * 0.72 - 0.1, ey * 0.72, r"$d\,\cos\,\theta$", ha="right", va="center", path_effects=halo, zorder=8)

    sx, sz = elevation(example)
    ax_side.plot([0, sx], [0, sz], color=INK, lw=0.7, zorder=5)
    ax_side.plot(sx, sz, "o", ms=2.8, mfc=CONTEXT, mec=INK, mew=0.5, zorder=7)
    ax_side.add_patch(Arc((0, 0), 3.2, 3.2, theta1=0, theta2=example["pitch_deg"], **arc))
    half_pitch = math.radians(example["pitch_deg"] / 2)
    ax_side.text(1.82 * math.cos(half_pitch), 1.82 * math.sin(half_pitch), r"$\theta$", ha="center", va="center",
                 path_effects=halo, zorder=8)
    ax_side.text(sx * 0.42, sz * 0.42 + 0.1, "$d$", ha="center", va="bottom", path_effects=halo, zorder=8)

    handles = [Line2D([], [], ls="none", marker="o", ms=2.4, mfc=CLOSEUP, mec="none", label="Close-up"),
               Line2D([], [], ls="none", marker="o", ms=3.0, mfc=CONTEXT, mec="none", label="Wider context")]
    fig.legend(handles=handles, loc="center", frameon=False, ncol=2, handletextpad=0.1, borderaxespad=0.0,
               borderpad=0.0, columnspacing=1.6,
               bbox_to_anchor=(0.5, (y_legend + legend_h / 2) / fig_h))

    # The key lights in the same two kinds of view, in metres from the target a_L (Eq. light_placement).
    reach = max(light["distance_m"] for light in lights)
    reach = math.ceil(reach * 2 + 0.2) / 2
    styles = {regime: (colour, marker) for regime, _, colour, marker in REGIMES}

    def section(light):
        """Radius r at incidence alpha from the normal. Azimuth is folded away, so both are exact."""
        alpha = math.radians(light["incidence_deg"])
        return light["distance_m"] * math.cos(alpha), light["distance_m"] * math.sin(alpha)

    def front(light):
        """Eq. (light_placement) seen straight on: the angle around a_L is azimuth."""
        alpha, beta = math.radians(light["incidence_deg"]), math.radians(light["azimuth_deg"])
        in_plane = light["distance_m"] * math.sin(alpha)
        return in_plane * math.cos(beta), in_plane * math.sin(beta)

    def scatter(axis, project):
        for regime, _, colour, marker in REGIMES:
            chosen = [project(light) for light in lights if light["regime"] == regime]
            axis.scatter(*zip(*chosen), s=4.0, color=colour, marker=marker, lw=0.15, edgecolors="white", zorder=3)

    # One draw carries the symbols in both views, picked by a rule.
    target = min((light for light in lights if light["regime"] == "window_raking"),
                 key=lambda light: abs(light["azimuth_deg"] - 40) + abs(light["incidence_deg"] - 66)
                 + 8 * abs(light["distance_m"] - 3.0))
    target_colour = styles[target["regime"]][0]

    # (c) Side section: the wall on the left, as in (b), and the normal running out from it.
    ax_c = axes(left, y_light, plan_w, light_h)
    style(ax_c)
    ax_c.set_xlim(0, reach * plan_w / light_h)
    ax_c.set_ylim(0, reach)
    ax_c.set_title("(c) Key lights, side view", pad=3)
    ax_c.set_xlabel("Out from the wall (m)", labelpad=1.5)
    ax_c.set_ylabel("Along the wall plane (m)", labelpad=1.5)
    ax_c.plot((0, 0), (0, reach), color=WALL, lw=1.6, solid_capstyle="butt", clip_on=False, zorder=2)
    ax_c.text(0.12, reach, "Wall", ha="left", va="top", color=INK, zorder=8, path_effects=halo)
    ax_c.text(reach * plan_w / light_h, 0.1, "Wall\nnormal", ha="right", va="bottom", color=INK, linespacing=1.05, zorder=8, path_effects=halo)
    scatter(ax_c, section)
    tx, ty = section(target)
    ax_c.plot((0, tx), (0, ty), color=INK, lw=0.7, zorder=5)
    ax_c.plot(tx, ty, "o", ms=3.0, mfc=target_colour, mec=INK, mew=0.5, zorder=7)
    ax_c.add_patch(Arc((0, 0), 2.6, 2.6, theta1=0, theta2=target["incidence_deg"], color=INK, lw=0.5, zorder=6))
    half = math.radians(target["incidence_deg"] / 2)
    ax_c.text(1.58 * math.cos(half), 1.58 * math.sin(half), r"$\alpha$", ha="center", va="center", path_effects=halo, zorder=8)
    ax_c.text(tx * 0.72 - 0.08, ty * 0.72 + 0.1, "$r$", ha="right", va="bottom", path_effects=halo, zorder=8)

    # (d) Front view: the panel is the wall itself, seen from the room.
    ax_d = axes(left + plan_w + gap, y_light, cam_w, light_h)
    style(ax_d)
    ax_d.set_xlim(-reach, reach)
    ax_d.set_ylim(-reach, reach)
    ax_d.set_xticks((-4, -2, 0, 2, 4))
    ax_d.set_yticks((-4, -2, 0, 2, 4))
    ax_d.set_facecolor("#ECEAE6")
    ax_d.set_title("(d) Key lights, front view", pad=3)
    ax_d.set_xlabel("Along the wall (m)", labelpad=1.5)
    ax_d.set_ylabel("Up the wall (m)", labelpad=1.5)
    ax_d.plot((-reach, reach), (0, 0), color=MUTED, lw=0.4, zorder=1)
    ax_d.plot((0, 0), (-reach, reach), color=MUTED, lw=0.4, zorder=1)
    ax_d.text(-reach + 0.15, reach - 0.15, "Wall", ha="left", va="top", color=INK, zorder=8)
    scatter(ax_d, front)
    tx, ty = front(target)
    ax_d.plot((0, tx), (0, ty), color=INK, lw=0.7, zorder=5)
    ax_d.plot(tx, ty, "o", ms=3.0, mfc=target_colour, mec=INK, mew=0.5, zorder=7)
    ax_d.add_patch(Arc((0, 0), 3.0, 3.0, theta1=0, theta2=target["azimuth_deg"], color=INK, lw=0.5, zorder=6))
    half = math.radians(target["azimuth_deg"] / 2)
    ax_d.text(1.95 * math.cos(half), 1.95 * math.sin(half), r"$\beta$", ha="center", va="center", path_effects=halo, zorder=8)

    handles = [Line2D([], [], ls="none", marker=marker, ms=3.2, mfc=colour, mec="white", mew=0.2, label=label)
               for _, label, colour, marker in REGIMES]
    fig.legend(handles=handles, loc="lower center", frameon=False, ncol=3, handletextpad=0.1, borderaxespad=0.0,
               borderpad=0.0, columnspacing=1.2, labelspacing=0.3, bbox_to_anchor=(0.5, 0.01))

    fig.savefig(FIGURE, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)
    print(FIGURE, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("sample", "check", "plot"))
    args = parser.parse_args()
    {"sample": sample, "check": check, "plot": plot}[args.stage]()


if __name__ == "__main__":
    main()
