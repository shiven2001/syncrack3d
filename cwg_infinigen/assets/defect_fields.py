# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Defect shapes rebuilt as shader fields, so they can live inside a host wall.

The plane factories paint each defect onto its own card. A card cannot show a
recess - the opaque wall sits right behind it - so a groove modelled that way
reads as a raised welt, and the card's own silhouette and texture give it away
at grazing incidence. Evaluating the same shape as a field inside the host
wall's material and letting Cycles displace it removes both problems: the
defect is real geometry on the wall itself, and there is no second surface.

Each ``build_*_field`` emits the identical shape its factory draws, from the
same parameters, against a coordinate socket the caller supplies. One socket
feeds both the displacement and the AOV label, so the mask cannot disagree with
the geometry - which matters here, because these defects are meant to become
invisible under some lighting while staying correctly annotated.
"""

import numpy as np

from cwg_infinigen.core.util.math import FixedSeed

# --------------------------------------------------------------------------- #
# small node helpers, shared by the three builders
# --------------------------------------------------------------------------- #


def _nodes(nt):
    def node(kind, **props):
        n = nt.nodes.new(kind)
        for key, val in props.items():
            setattr(n, key, val)
        return n

    def math(op, a=None, b=None, clamp=False):
        m = node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, operand in enumerate((a, b)):
            if operand is None:
                continue
            if hasattr(operand, "is_linked"):
                nt.links.new(operand, m.inputs[i])
            else:
                m.inputs[i].default_value = float(operand)
        return m.outputs["Value"]

    def rng(value, x0, x1, y0=0.0, y1=1.0, interp="LINEAR"):
        mr = node("ShaderNodeMapRange", interpolation_type=interp, clamp=True)
        mr.inputs["From Min"].default_value = x0
        mr.inputs["From Max"].default_value = x1
        mr.inputs["To Min"].default_value = y0
        mr.inputs["To Max"].default_value = y1
        nt.links.new(value, mr.inputs["Value"])
        return mr.outputs["Result"]

    def noise(vector, scale, detail=6.0, roughness=0.55, ntype="FBM"):
        n = node("ShaderNodeTexNoise", noise_dimensions="3D", noise_type=ntype)
        if hasattr(n, "normalize"):
            n.normalize = True
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = roughness
        nt.links.new(vector, n.inputs["Vector"])
        return n.outputs["Fac"]

    return node, math, rng, noise


# --------------------------------------------------------------------------- #
# crack
# --------------------------------------------------------------------------- #


def sample_crack_morphology(seed: int, half: float = 0.14, profile: str = "legacy") -> dict:
    """Mirror of the morphology draws at the top of create_crack_material.

    Replayed rather than shared because those draws sit inside a 200-line block
    interleaved with per-orientation colour work, and splitting it would change
    the seed contract for every crack already generated. tests/test_defect_field
    _drift.py records the factory's RNG calls and asserts this reproduces them,
    so an edit to either side fails the suite instead of silently diverging.
    """
    if profile not in ("legacy", "major", "mixed"):
        raise ValueError("crack profile must be legacy, major or mixed")
    from cwg_infinigen.core.util.math import int_hash
    profile_rng = np.random.default_rng(int_hash((seed, "major_crack_v1")))
    major = profile == "major" or (profile == "mixed" and profile_rng.random() < 0.5)
    with FixedSeed(seed):
        style = str(
            np.random.choice(
                ["hairline", "hairline_fork", "moderate", "branched", "open", "double_run"],
                p=[0.14, 0.16, 0.28, 0.20, 0.16, 0.06],
            )
        )
        if major:
            style = str(profile_rng.choice(["major_open", "major_branched"]))
        if style in ("major_open", "major_branched"):
            # Radius in metres: full untapered openings 3–24 mm before scaling.
            # An appearance profile, not a structural severity diagnosis.
            w_min = np.random.uniform(0.0015, 0.0035)
            w_max = np.random.uniform(0.0050, 0.0120)
            n_kinks = int(np.random.randint(2, 5))
            n_forks = int(np.random.randint(3, 7)) if style == "major_branched" else 1
            y_split_enabled = style == "major_branched"
        elif style == "hairline":
            w_min = np.random.uniform(0.00018, 0.00040)
            w_max = np.random.uniform(0.00050, 0.0016)
            n_kinks = int(np.random.randint(0, 2))
            n_forks = 0
            y_split_enabled = False
        elif style == "hairline_fork":
            w_min = np.random.uniform(0.00022, 0.00050)
            w_max = np.random.uniform(0.00070, 0.0022)
            n_kinks = int(np.random.randint(0, 2))
            n_forks = int(np.random.randint(1, 3))
            y_split_enabled = bool(np.random.rand() < 0.55)
        elif style == "moderate":
            w_min = np.random.uniform(0.00028, 0.00065)
            w_max = np.random.uniform(0.0011, 0.0032)
            n_kinks = int(np.random.randint(1, 3))
            n_forks = int(np.random.randint(2, 5))
            y_split_enabled = bool(np.random.rand() < 0.70)
        elif style == "branched":
            w_min = np.random.uniform(0.00032, 0.00080)
            w_max = np.random.uniform(0.0016, 0.0042)
            n_kinks = int(np.random.randint(1, 4))
            n_forks = int(np.random.randint(3, 7))
            y_split_enabled = True
        elif style == "open":
            # 0.8-6 mm mouth. Wider than a hairline, still a finishing crack.
            w_min = np.random.uniform(0.00080, 0.0018)
            w_max = np.random.uniform(0.0025, 0.0060)
            n_kinks = int(np.random.randint(1, 4))
            n_forks = int(np.random.randint(1, 4))
            y_split_enabled = True
        else:
            w_min = np.random.uniform(0.00018, 0.00045)
            w_max = np.random.uniform(0.00055, 0.0018)
            n_kinks = int(np.random.randint(0, 2))
            n_forks = int(np.random.randint(0, 2))
            y_split_enabled = False

        w_fork_floor = np.random.uniform(0.00016, 0.00040)
        w_lip = np.random.uniform(0.0006, 0.0020)
        h_lip = np.random.uniform(0.0004, 0.0014)
        # Depth of the groove, in metres. Darkness has to come from the slot
        # seeing little of the hemisphere, not from a painted-on dark albedo.
        # A 5-17 mm U-channel leaves a floor that is fully lit; a 0.5-4 mm V
        # at 3-7 mm deep is steep enough that the camera mostly sees the
        # shadowed walls. MUST stay identical to create_crack_material.
        crack_depth = np.random.uniform(0.0030, 0.0070)
        if major:
            crack_depth = float(profile_rng.uniform(0.008, 0.018))
            w_fork_floor = float(profile_rng.uniform(0.0004, 0.0012))

        crack_angle = np.random.uniform(0, 2 * np.pi)
        crack_offset = np.random.uniform(-0.08, 0.08)
        np.random.rand()
        np.random.uniform(0.35, 1.0)

        wander_freq = np.random.uniform(0.9, 4.5)
        wander_amp = np.random.uniform(0.05, 0.22)
        jag_freq = np.random.uniform(16.0, 72.0)
        jag_amp = np.random.uniform(0.004, 0.028)
        micro_freq = np.random.uniform(55.0, 180.0)
        micro_amp = np.random.uniform(0.0005, 0.0032)
        width_freq = np.random.uniform(3.5, 16.0)

        kink_specs = [
            {
                "v": float(np.random.uniform(-0.18, 0.18)),
                "slope": float(
                    np.random.uniform(0.15, 1.05)
                    * (1.0 if np.random.rand() > 0.5 else -1.0)
                ),
            }
            for _ in range(n_kinks)
        ]

        fork_specs = []
        if n_forks > 0 and y_split_enabled:
            y_split = float(np.random.uniform(-0.10, 0.10))
            y_slope = float(np.random.uniform(0.45, 2.8))
            y_dir = 1.0 if np.random.rand() > 0.5 else -1.0
            fork_specs.extend(
                [
                    {
                        "split": y_split,
                        "slope": y_slope,
                        "length": float(np.random.uniform(0.10, 0.38)),
                        "direction": y_dir,
                        "width_scale": float(np.random.uniform(0.35, 0.85)),
                    },
                    {
                        "split": y_split,
                        "slope": -y_slope * float(np.random.uniform(0.25, 0.85)),
                        "length": float(np.random.uniform(0.06, 0.24)),
                        "direction": y_dir,
                        "width_scale": float(np.random.uniform(0.20, 0.60)),
                    },
                ]
            )
            n_forks = max(n_forks - 2, 0)

        split_cursor = float(np.random.uniform(-0.12, 0.12))
        for _ in range(n_forks):
            split_cursor += float(np.random.uniform(0.04, 0.14)) * (
                1.0 if np.random.rand() > 0.5 else -1.0
            )
            fork_specs.append(
                {
                    "split": float(np.clip(split_cursor, -0.22, 0.22)),
                    "slope": float(
                        np.random.uniform(0.35, 2.6)
                        * (1.0 if np.random.rand() > 0.5 else -1.0)
                    ),
                    "length": float(np.random.uniform(0.05, 0.32)),
                    "direction": 1.0 if np.random.rand() > 0.5 else -1.0,
                    "width_scale": float(np.random.uniform(0.18, 0.75)),
                }
            )

    return dict(
        style=style,
        w_min=float(w_min),
        w_max=float(w_max),
        w_fork_floor=float(w_fork_floor),
        w_lip=float(w_lip),
        h_lip=float(h_lip),
        crack_depth=float(crack_depth),
        crack_angle=float(crack_angle),
        crack_offset=float(crack_offset),
        wander_freq=float(wander_freq),
        wander_amp=float(wander_amp),
        jag_freq=float(jag_freq),
        jag_amp=float(jag_amp),
        micro_freq=float(micro_freq),
        micro_amp=float(micro_amp),
        width_freq=float(width_freq),
        kink_specs=kink_specs,
        fork_specs=fork_specs,
        half_u=half,
        half_v=half,
    )


def adapt_crack_morphology(p, *, width_scale=1.0, jitter_scale=1.0,
                           depth_scale=1.0, fit_path=False):
    """Explicit injected-field ablation; no new random draws or default drift.

    Width changes physical geometry AND its AOV, not just annotation support.
    Jitter changes high-frequency roughness while retaining the coarse path,
    kinks, and connected forks. The legacy card factory is left unchanged.
    """
    if not 0.25 <= width_scale <= 4.0 or not 0 <= jitter_scale <= 2.0:
        raise ValueError("width_scale must be .25..4 and jitter_scale 0..2")
    if not 0.05 <= depth_scale <= 2:
        raise ValueError("depth_scale must be .05..2")
    adjusted = dict(p)
    for key in ("w_min", "w_max", "w_fork_floor", "w_lip"):
        adjusted[key] *= width_scale
    for key in ("jag_amp", "micro_amp"):
        adjusted[key] *= jitter_scale
    adjusted["crack_depth"] *= depth_scale
    if fit_path:
        # Noise lies in [-1, 1] after recentering/scaling. Bound the main
        # centerline including kinks, so small cards cannot clip it into islands.
        bound = sum(abs(adjusted[k]) for k in
                    ("wander_amp", "jag_amp", "micro_amp", "crack_offset"))
        bound += sum(max(0, p["half_v"] - k["v"]) * abs(k["slope"])
                     for k in p["kink_specs"])
        scale = min(1.0, 0.75 * p["half_u"] / max(bound, 1e-9))
        for key in ("wander_amp", "jag_amp", "micro_amp", "crack_offset"):
            adjusted[key] *= scale
        adjusted["kink_specs"] = [dict(k, slope=k["slope"] * scale) for k in p["kink_specs"]]
    return adjusted


def adapt_crack_path(p, seed, mode="original"):
    """Opt-in dimension-normalized broad bends; no global RNG consumption.

    The old noise frequency is in metres, so a small card can sample a nearly
    straight portion. Knots span the actual card, with independently bounded
    fine roughness. This is a morphology hypothesis, not a fracture simulation.
    """
    if mode == "original":
        return p
    if mode not in ("meander", "angular"):
        raise ValueError(f"Unknown crack path mode: {mode}")
    rng = np.random.default_rng(int(seed) ^ 0x4D45414E)
    if mode == "angular":
        # Uneven piecewise-linear turns, not smooth periodic strands. Keep
        # short secondary branches sparse so they do not duplicate the trunk.
        count = int(rng.integers(10, 22))
        vs = np.r_[-p["half_v"], np.sort(rng.uniform(-.94, .94, count-2))*p["half_v"], p["half_v"]]
        x = float(rng.uniform(-.15, .15)*p["half_u"])
        knots = []
        for v in vs:
            x = float(np.clip(x + rng.normal(0, .12)*p["half_u"], -.45*p["half_u"], .45*p["half_u"]))
            knots.append([float(v), x])
        forks = [dict(s, length=min(s["length"], float(rng.uniform(.25, .6))*p["half_v"]))
                 for s in p["fork_specs"][:int(rng.choice([0, 1, 2], p=[.45, .45, .10]))]]
        return dict(p, path_mode=mode, path_knots=knots, path_interpolation="LINEAR",
                    fork_specs=forks, kink_specs=[], crack_offset=0., wander_amp=0.,
                    jag_amp=.012*p["half_u"], jag_freq=35/max(2*p["half_v"], .01),
                    micro_amp=.003*p["half_u"])
    count = int(rng.integers(5, 9))
    x = float(rng.uniform(-.18, .18) * p["half_u"])
    knots = []
    for v in np.linspace(-p["half_v"], p["half_v"], count):
        x = float(np.clip(x + rng.uniform(-.28, .28)*p["half_u"], -.42*p["half_u"], .42*p["half_u"]))
        knots.append([float(v), x])
    return dict(p, path_mode=mode, path_knots=knots, kink_specs=[], crack_offset=0.,
                wander_amp=0., jag_amp=min(p["jag_amp"], .015*p["half_u"]),
                micro_amp=min(p["micro_amp"], .005*p["half_u"]))


def adapt_crack_branches(p, mode="original"):
    """Opt-in bounded, curved forks; preserve RNG and the parent crack field.

    Old forks can start outside a small placement card or run into its border.
    Keep split points inside the parent span and reserve lateral room for the
    full curved branch plus its maximum half-width. Requires a fitted parent.
    The shader uses the same returned parameters for geometry and coverage.
    """
    if mode == "original":
        return p
    if mode not in ("bounded_curved", "local_curved"):
        raise ValueError(f"Unknown crack branch mode: {mode}")
    bound = sum(abs(p[k]) for k in
                ("wander_amp", "jag_amp", "micro_amp", "crack_offset"))
    bound += sum(max(0, p["half_v"] - k["v"]) * abs(k["slope"])
                 for k in p["kink_specs"])
    if p.get("path_knots"):
        bound += max(abs(k[1]) for k in p["path_knots"])
    margin = .93 * p["half_u"] - bound - max(p["w_max"], p["w_fork_floor"])
    if margin <= 0:
        raise ValueError("Curved branches require fitted parent with room for width")
    forks = []
    for spec in p["fork_specs"]:
        split = float(np.clip(spec["split"], -.55 * p["half_v"], .55 * p["half_v"]))
        available = .90 * p["half_v"] - split * spec["direction"]
        length = min(spec["length"], available)
        curvature = .18
        slope = float(np.clip(spec["slope"], -margin / (length * (1 + curvature)),
                              margin / (length * (1 + curvature))))
        if mode == "local_curved":
            # Keep the sampled branching angle. The shader bounds the departure
            # using the available space next to its actual parent coordinate.
            slope = spec["slope"]
        forks.append(dict(spec, split=split, length=length, slope=slope,
                          curvature=curvature))
    return dict(p, fork_specs=forks, branch_mode=mode)


def build_crack_field(nt, coord_socket, p, u_idx="X", v_idx="Y", out=None):
    """Emit the crack profile: 1 in the channel, 0 off it, soft at the edges.

    ``out`` (optional dict) receives the u/v, centreline and width sockets of the
    main run, for callers that need which side of the crack a point is on.
    """
    node, math, rng, _ = _nodes(nt)
    sep = node("ShaderNodeSeparateXYZ")
    nt.links.new(coord_socket, sep.inputs["Vector"])
    u, v = sep.outputs[u_idx], sep.outputs[v_idx]

    def noise1d(coord, scale, detail, seed_off):
        """1-D value noise along the run, centred on zero."""
        c = node("ShaderNodeCombineXYZ")
        nt.links.new(math("MULTIPLY", coord, scale), c.inputs["X"])
        c.inputs["Y"].default_value = float(seed_off)
        n = node("ShaderNodeTexNoise", noise_dimensions="3D")
        n.inputs["Scale"].default_value = 1.0
        n.inputs["Detail"].default_value = float(detail)
        n.inputs["Roughness"].default_value = 0.55
        nt.links.new(c.outputs["Vector"], n.inputs["Vector"])
        return math("SUBTRACT", n.outputs["Fac"], 0.5)

    # centreline: coarse wander, then jag, then fine jitter
    meander = math(
        "MULTIPLY", noise1d(v, p["wander_freq"], 3.0, 11.3), p["wander_amp"] * 2.0
    )
    jag = math("MULTIPLY", noise1d(v, p["jag_freq"], 5.0, 27.7), p["jag_amp"] * 2.0)
    micro = math(
        "MULTIPLY", noise1d(v, p["micro_freq"], 7.0, 41.1), p["micro_amp"] * 2.0
    )
    path = math("ADD", math("ADD", meander, jag), micro)
    if p.get("crack_offset"):
        path = math("ADD", path, float(p["crack_offset"]))
    for kink in p.get("kink_specs", []):
        path = math(
            "ADD",
            path,
            math(
                "MULTIPLY",
                math("MAXIMUM", math("SUBTRACT", v, float(kink["v"])), 0.0),
                float(kink["slope"]),
            ),
        )

    if p.get("path_knots"):
        knots = p["path_knots"]
        path = math("ADD", knots[0][1], math("ADD", jag, micro))
        for (v0, x0), (v1, x1) in zip(knots, knots[1:]):
            path = math("ADD", path, rng(v, v0, v1, 0., x1 - x0, p.get("path_interpolation", "SMOOTHSTEP")))

    w_eff = rng(
        math("ADD", noise1d(v, p["width_freq"], 4.0, 63.9), 0.5),
        0.30,
        0.75,
        p["w_min"],
        p["w_max"],
        interp="SMOOTHSTEP",
    )

    # A crack ends at a tip: the channel narrows to nothing while staying at
    # full depth to the last millimetre. It does not get shallower and stay
    # wide, which is what the border fade below used to do on its own and what
    # read as the line gradienting out at both ends.
    #
    # Applied to w_eff rather than to the profile, so the forks - which scale
    # their own width off it - taper with the main run instead of ending
    # abruptly beside a tapering parent.
    # The floor is 0.22, not 0.05. w_eff is already modulated along the run by
    # width_freq noise between w_min and w_max, so a taper that goes near zero
    # drives the product under the displacement threshold at every noise minimum
    # and the last stretch of crack renders as a row of detached dashes rather
    # than a tapering line. 0.22 over the outer fifth still reads as a tip.
    half_v = float(p.get("half_v", 0.14))
    w_eff = math(
        "MULTIPLY",
        w_eff,
        rng(math("ABSOLUTE", v), half_v * 0.80, half_v * 0.98, 1.0, 0.22, "SMOOTHSTEP"),
    )

    if out is not None:
        out.update(u=u, v=v, path=path, w_eff=w_eff)

    def channel(path_socket, width_socket, gate=None):
        d = math("ABSOLUTE", math("SUBTRACT", u, path_socket))
        # Linear V, matching create_crack_material's ramp(dist, 0, w, 1, 0).
        # (1-(d/w)^2)^0.55 was a wide U: the floor stayed in full light.
        prof = math("SUBTRACT", 1.0, math("DIVIDE", d, width_socket), clamp=True)
        return prof if gate is None else math("MULTIPLY", prof, gate)

    prof = channel(path, w_eff)

    # forks peel off at a point, drift away, and taper to nothing
    for spec in p["fork_specs"]:
        along = math(
            "MAXIMUM",
            math("MULTIPLY", math("SUBTRACT", v, spec["split"]), spec["direction"]),
            0.0,
        )
        fade = rng(along, spec["length"] * 0.55, spec["length"], 1.0, 0.0, "SMOOTHSTEP")
        taper = rng(along, 0.0, max(spec["length"], 1e-6), 1.0, 0.12)
        departure = math("MULTIPLY", along, spec["slope"])
        if spec.get("curvature", 0):
            phase = math("MULTIPLY", math("DIVIDE", along, spec["length"]), float(np.pi))
            bend = math("ADD", 1.0, math("MULTIPLY", math("SINE", phase), spec["curvature"]))
            departure = math("MULTIPLY", departure, bend)
        if p.get("branch_mode") == "local_curved":
            direction = 1.0 if spec["slope"] >= 0 else -1.0
            edge = .93 * p["half_u"] - max(p["w_max"], p["w_fork_floor"])
            room = math("MAXIMUM", math("SUBTRACT", edge, math("MULTIPLY", path, direction)), 1e-6)
            # d/(1+|d|/room): derivative 1 at the junction, smooth asymptote
            # inside the card, no hard clipping or global angle flattening.
            departure = math("DIVIDE", departure, math("ADD", 1.0,
                math("DIVIDE", math("ABSOLUTE", departure), room)))
        path_f = math("ADD", path, departure)
        w_f = math(
            "MAXIMUM",
            math("MULTIPLY", math("MULTIPLY", w_eff, spec["width_scale"]), taper),
            p["w_fork_floor"],
        )
        prof = math("MAXIMUM", prof, channel(path_f, w_f, gate=fade))

    # Kill the profile at the patch border so the groove never ends in a cliff.
    # This is now a 6% band rather than 20%: the width taper above already
    # brings the run to a point, so a long depth ramp here only softened the
    # last few centimetres of a crack that had already ended.
    for axis, lim in ((u, p.get("half_u", 0.14)), (v, p.get("half_v", 0.14))):
        prof = math(
            "MULTIPLY",
            prof,
            rng(math("ABSOLUTE", axis), lim * 0.94, lim, 1.0, 0.0, "SMOOTHSTEP"),
        )
    return prof
