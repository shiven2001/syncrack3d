"""Crack placement on one wall, seen front-on, from the generator's own draws.

scene.json keeps only final centres, so `trace` runs the unified recipe up to the
junction step and logs every candidate: the same random stream is replayed through
the sampler's rules to record rejections with their reason, and the replay must
reproduce the generator's accepted centres exactly. `plot` draws one trace; no
position in the figure is placed by hand.

    # log candidates per seed -> outputs/placement_figure/traces/, prints a summary
    .venv/bin/python tools/render_placement_figure.py trace --seeds 0-99

    # -> outputs/placement_figure/crack_placement.png
    .venv/bin/python tools/render_placement_figure.py plot --seed 274

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "outputs" / "placement_figure"
TRACES = OUT / "traces"
FIGURE = OUT / "crack_placement.png"
FIGURE_SEED = 499  # both rejection kinds, a token beside a run, and one move to a window corner
COLUMN_WIDTH_IN = 3.297  # \columnwidth of cas-dc: 238.25 pt
FIGURE_DPI = 600
KEEPOUT_GROW = (0.3, 0.03)  # gamma = k_s * s_max + gamma_0, as in defects._sample_positions


class _StopBuild(Exception):
    """Everything the figure needs is known once the junction step has run."""


def _replay(rng, plan, width_m, height_m, count, keepout, near, near_fraction) -> dict:
    """defects._sample_positions on a copy of its stream, logging each candidate's fate."""
    from syncrack3d.defects import _EDGE_MARGIN_M

    half_u = max(width_m * 0.5 - _EDGE_MARGIN_M, 0.05)
    half_v_lo = _EDGE_MARGIN_M
    half_v_hi = max(height_m - _EDGE_MARGIN_M, half_v_lo + 0.05)
    grow = KEEPOUT_GROW[0] * plan.size_range_m[1] + KEEPOUT_GROW[1]

    def blocked(u, v):
        return any(abs(u - ku) < kw * 0.5 + grow and abs(v - kv) < kh * 0.5 + grow
                   for ku, kv, kw, kh in keepout)

    placed, beside_flags, rejected = [], [], []
    for token in range(count):
        beside = bool(near) and near_fraction > 0 and rng.random() < near_fraction
        for _ in range(80):
            if beside:
                nu, nv, nw, nh = rng.choice(near)
                off = rng.choice((-1, 1)) * rng.uniform(0.03, 0.15)
                if nh >= nw:
                    u, v = nu + off + math.copysign(nw * 0.5, off), rng.uniform(nv - nh * 0.5, nv + nh * 0.5)
                else:
                    u, v = rng.uniform(nu - nw * 0.5, nu + nw * 0.5), nv + off + math.copysign(nh * 0.5, off)
                if not (-half_u <= u <= half_u and half_v_lo <= v <= half_v_hi):
                    rejected.append({"token": token, "u": u, "v": v, "reason": "margin"})
                    continue
            else:
                u = rng.uniform(-half_u, half_u)
                v = rng.uniform(half_v_lo, half_v_hi)
            if keepout and blocked(u, v):
                rejected.append({"token": token, "u": u, "v": v, "reason": "keepout"})
                continue
            if all(math.hypot(u - pu, v - pv) >= plan.min_separation_m for pu, pv in placed):
                placed.append((u, v))
                beside_flags.append(beside)
                break
            rejected.append({"token": token, "u": u, "v": v, "reason": "spacing"})
        else:
            break
    return {"placed": placed, "beside": beside_flags, "rejected": rejected, "grow_m": grow}


def trace(seeds: str) -> None:
    import syncrack3d.build_dataset as bd
    import syncrack3d.defects as defects
    from syncrack3d.structures import window_trim

    TRACES.mkdir(parents=True, exist_ok=True)
    state: dict = {}
    real_sample = defects._sample_positions
    real_snap = bd.snap_to_junctions
    real_style = bd.wall_style

    def sample(rng, plan, width_m, height_m, count, keepout=(), near=(), near_fraction=0.0):
        copy = random.Random()
        copy.setstate(rng.getstate())
        log = _replay(copy, plan, width_m, height_m, count, keepout, near, near_fraction)
        placed = real_sample(rng, plan, width_m, height_m, count, keepout=keepout, near=near,
                             near_fraction=near_fraction)
        if placed != log["placed"]:
            raise RuntimeError("the replay left the generator's stream")
        state.update(log, keepout=[list(k) for k in keepout], near=[list(n) for n in near],
                     min_separation_m=plan.min_separation_m, size_range_m=list(plan.size_range_m),
                     near_fraction=near_fraction)
        return placed

    def snap(placements, structure_plans, spec, seed, fraction, min_separation_m=0.0):
        moved, records = real_snap(placements, structure_plans, spec, seed, fraction, min_separation_m)
        log = {
            "seed": seed, "width_m": spec.width_m, "height_m": spec.height_m,
            "margin_m": defects._EDGE_MARGIN_M, "style": state.get("style"),
            "tokens": [{"index": p.index, "size_m": p.size_m, "sampled": [p.u_m, p.v_m],
                        "final": [q.u_m, q.v_m], "beside_run": bool(flag)}
                       for p, q, flag in zip(placements, moved, state["beside"])],
            "junctions": records,
            "fixtures": [{"kind": s.kind, "u": s.u_m, "v": s.v_m, "w": s.width_m, "h": s.height_m,
                          **({"trim": {k: getattr(window_trim(s.seed), k)
                                       for k in ("horn", "drop", "overhang", "rise")}}
                             if s.kind == "window" else {})}
                         for s in structure_plans],
            **{k: state[k] for k in ("rejected", "keepout", "near", "min_separation_m",
                                     "size_range_m", "grow_m", "near_fraction")},
        }
        (TRACES / f"seed_{seed:05d}.json").write_text(json.dumps(log, indent=1) + "\n")
        reasons = [r["reason"] for r in log["rejected"]]
        print(f"TRACE {seed} {log['style']} W={spec.width_m:.2f} H={spec.height_m:.2f} "
              f"tokens={len(placements)} keepout_rej={reasons.count('keepout')} "
              f"spacing_rej={reasons.count('spacing')} "
              f"moves={[j['anchor'] for j in records]} "
              f"beside={sum(state['beside'])} fixtures={[s.kind for s in structure_plans]}", flush=True)
        raise _StopBuild

    def style(seed, args):
        name, wall = real_style(seed, args)
        state["style"] = name
        return name, wall

    defects._sample_positions = sample
    bd.snap_to_junctions = snap
    bd.wall_style = style
    sys.argv = ["build_dataset", "--recipe", "unified", "--seeds", seeds, "--output", str(OUT / "scenes"),
                "--no-render", "--keep-going"]
    bd.main()


def _anchor(token: dict, record: dict) -> tuple[float, float] | None:
    """The fixture point a junction move starts from (junctions.snap_to_junctions)."""
    if record["mode"] != "corner":
        return record["u_m"], record["v_m"]
    half = 0.42 * min(token["size_m"], 0.42)
    angle = math.radians(record["angle_deg"])
    return record["u_m"] - math.cos(angle) * half * 0.95, record["v_m"] - math.sin(angle) * half * 0.95


def plot(seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import patheffects
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch, Rectangle

    WALL, FIXTURE, RUN, TOKEN, MOVE = "#D55E00", "#CC3D94", "#8C8C8C", "#0072B2", "#009E73"
    INK, FONT_PT = "#1A1A1A", 6.5
    data = json.loads((TRACES / f"seed_{seed:05d}.json").read_text())
    width, height, margin = data["width_m"], data["height_m"], data["margin_m"]
    spacing, grow = data["min_separation_m"], data["grow_m"]
    half = width / 2
    halo = [patheffects.withStroke(linewidth=1.4, foreground="white")]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_PT, "mathtext.fontset": "dejavusans"})
    pad = 0.03
    plot_w, left, right = width + 2 * pad, 0.11, 0.01
    axes_w_in = COLUMN_WIDTH_IN * (1 - left - right)
    axes_h_in = axes_w_in * (height + 2 * pad) / plot_w
    bottom_in, legend_in, top_in = 0.28, 0.53, 0.03
    fig_h = axes_h_in + bottom_in + legend_in + top_in
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, fig_h))
    ax = fig.add_axes((left, (bottom_in + legend_in) / fig_h, 1 - left - right, axes_h_in / fig_h))
    ax.set_xlim(-half - pad, half + pad)
    ax.set_ylim(-pad, height + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("$u$ (m)", labelpad=1.5)
    ax.set_ylabel("$v$ (m)", labelpad=1.5)
    ax.tick_params(length=2, width=0.5, pad=1.5, labelsize=FONT_PT - 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    def rect(cx, cy, w, h, **style):
        return ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, **style))

    def dimension(start, end, text, at, **style):
        ax.annotate("", xy=end, xytext=start, zorder=9,
                    arrowprops=dict(arrowstyle="<|-|>,head_width=0.09,head_length=0.16", color=INK, lw=0.5,
                                    shrinkA=0, shrinkB=0))
        ax.text(*at, text, color=INK, zorder=9, path_effects=halo, **style)

    wall = rect(0, height / 2, width, height, fc=(*matplotlib.colors.to_rgb(WALL), 0.07), ec="none", zorder=0)

    # Service runs are not keep-outs: plain bands, candidates may land on them.
    for fixture in data["fixtures"]:
        if fixture["kind"] in ("door", "window", "vent", "access_panel", "utility_box"):
            continue
        rect(fixture["u"], fixture["v"], fixture["w"], fixture["h"], fc="#E4E2E2", ec=RUN, lw=0.4, zorder=1)
        ax.text(-half + 0.34, fixture["v"], fixture["kind"].replace("_", " ").capitalize(), ha="left", va="center",
                color=INK, zorder=2)

    # Keep-outs: the rectangle grown by gamma is what rejects a centre.
    for ku, kv, kw, kh in data["keepout"]:
        zone = rect(ku, kv, kw + 2 * grow, kh + 2 * grow, fc=(*matplotlib.colors.to_rgb(FIXTURE), 0.20),
                    ec=FIXTURE, lw=0.5, ls=(0, (3, 1.5)), zorder=2)
        zone.set_clip_path(wall)  # a zone can reach past the floor line; the wall ends there
    for fixture in data["fixtures"]:
        if fixture["kind"] not in ("door", "window"):
            continue
        u, v, w, h = (fixture[k] for k in "uvwh")
        rect(u, v, w, h, fc="white", ec=FIXTURE, lw=0.7, zorder=3)
        if "trim" in fixture:
            trim = fixture["trim"]
            head = (w + 2 * trim["overhang"], trim["rise"])
            sill = (w + 2 * trim["horn"], trim["drop"])
            rect(u, v + h / 2 + head[1] / 2, *head, fc="white", ec=FIXTURE, lw=0.7, zorder=3)
            rect(u, v - h / 2 - sill[1] / 2, *sill, fc="white", ec=FIXTURE, lw=0.7, zorder=3)
        ax.text(u, v - h * 0.18, fixture["kind"].capitalize(), ha="center", va="center", color=INK, zorder=4)

    # Wall and margin over the bands, so both read as continuous lines.
    rect(0, height / 2, width, height, fc="none", ec=WALL, lw=1.0, zorder=5, clip_on=False)
    rect(0, height / 2, width - 2 * margin, height - 2 * margin, fc="none", ec=WALL, lw=0.7,
         ls=(0, (4, 2)), zorder=5)

    for token in data["tokens"]:
        ax.add_patch(Circle(token["sampled"], spacing, fc="none", ec=TOKEN, lw=0.6, ls=(0, (2, 1.5)), zorder=6))
    moved = {j["index"]: j for j in data["junctions"]}
    for token in data["tokens"]:
        sampled, final = token["sampled"], token["final"]
        if token["index"] in moved:
            record = moved[token["index"]]
            ax.plot(*sampled, "o", ms=3.4, mfc="white", mec=TOKEN, mew=0.8, zorder=8)
            ax.annotate("", xy=final, xytext=sampled, zorder=7,
                        arrowprops=dict(arrowstyle="-|>,head_width=0.14,head_length=0.28", color=MOVE, lw=0.9,
                                        shrinkA=2.0, shrinkB=2.0, connectionstyle="arc3,rad=-0.12"))
            anchor = _anchor(token, record)
            ax.plot(*anchor, "D", ms=2.8, mfc=MOVE, mec="white", mew=0.3, zorder=8)
            ax.text(anchor[0] - 0.04, anchor[1] + 0.05, r"$\mathbf{a}$", ha="right", va="bottom", color=INK,
                    zorder=9, path_effects=halo)
        ax.plot(*final, "o", ms=3.4, mfc=TOKEN, mec="white", mew=0.4, zorder=8)

    for candidate in data["rejected"]:
        ax.plot(candidate["u"], candidate["v"], "x", ms=4.2, mew=1.0, color=INK, zorder=8,
                path_effects=[patheffects.withStroke(linewidth=1.8, foreground="white")])

    # Dimensions, each in free wall.
    dimension((-half, 1.45), (-half + margin, 1.45), "$m$", (-half + margin + 0.04, 1.45), ha="left", va="center")
    door = next(k for k in data["keepout"] if k[0] > 0)
    edge = door[0] - door[2] / 2
    dimension((edge - grow, 1.62), (edge, 1.62), r"$\gamma$", (edge - grow - 0.04, 1.62), ha="right", va="center")
    still = next(t for t in data["tokens"] if t["index"] not in moved and t["sampled"][1] < 1.2)
    cu, cv = still["sampled"]
    dimension((cu, cv), (cu, cv + spacing), r"$d_{\min}$", (cu - 0.04, cv + 0.15), ha="right", va="center")

    handles = [
        Line2D([], [], color=WALL, lw=0.7, ls=(0, (4, 2)), label="Wall margin $m$"),
        Patch(fc=(*matplotlib.colors.to_rgb(FIXTURE), 0.20), ec=FIXTURE, lw=0.5, ls=(0, (3, 1.5)),
              label=r"Fixture keep-out, grown by $\gamma$"),
        Patch(fc="#E4E2E2", ec=RUN, lw=0.4, label="Service run (no keep-out)"),
        Line2D([], [], color=TOKEN, lw=0.6, ls=(0, (2, 1.5)), label=r"Spacing radius $d_{\min}$"),
        Line2D([], [], ls="none", marker="o", ms=3.4, mfc=TOKEN, mec="white", mew=0.4, label="Accepted centre"),
        Line2D([], [], ls="none", marker="x", ms=4.2, mew=1.0, color=INK, label="Rejected candidate"),
        Line2D([], [], ls="none", marker="o", ms=3.4, mfc="white", mec=TOKEN, mew=0.8, label="Centre before move"),
        Line2D([], [], color=MOVE, lw=0.9, marker=">", ms=3, label="Move to window corner"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=FONT_PT,
               bbox_to_anchor=(0.5, 0.0), borderaxespad=0.25, borderpad=0.1, labelspacing=0.3,
               columnspacing=1.2, handlelength=1.8, handleheight=0.8, handletextpad=0.5)

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)
    print(FIGURE, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("trace", "plot"))
    parser.add_argument("--seeds", default="0-99")
    parser.add_argument("--seed", type=int, default=FIGURE_SEED)
    args = parser.parse_args()
    if args.stage == "trace":
        trace(args.seeds)
    else:
        plot(args.seed)


if __name__ == "__main__":
    main()
