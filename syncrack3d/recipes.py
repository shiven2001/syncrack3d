"""The dataset recipe for build_dataset.py --recipe unified.

build_dataset parses the recipe first and the command line after it, so any
flag given explicitly overrides the recipe.

unified is the v1.30 dataset. Each wall draws a crack style (standard, wide or
crazing; STYLES) that sets only what defines that kind of crack. Materials are
the 10 unified categories, drawn uniformly. Every wall is an indoor room wall
with fixtures, at most one prop, and is cracked. Half of the crack tokens move
to a fixture corner or edge. Crack edges use the strong profile, and a crack
path has no segment steep or short enough to pinch the groove.
"""

COMMON = ("--substrate paint --max-props 1 --resolution 640 --samples 64 --wall-normal-mode explicit "
          "--dicing-rate .75 --exposure-ev-range .4 1.3 --closeup-fill .6 1.5 --closeup-aim-jitter .3 "
          "--preserve-closeup-scale --closeups-per-defect 2 --crack-fit-path --crack-label-mode support "
          "--crack-path-mode angular --crack-branch-mode local_curved --wall-materials diverse "
          "--awb-strength 0 .5 --crack-layered-fraction .7")

# v2: one recipe for everything. Each wall draws a crack style; the style sets
# only what defines that kind of crack (and its setting), everything else -
# capture, rendering, materials, clean-wall policy - is shared.
STYLES = {
    "standard": dict(crack_profile="standard", min_defects=2, max_defects=3, defect_size_range=(.20, .45),
                   crack_width_scale=1.3, crack_depth_scale=.5, structures="off"),
    "wide": dict(crack_profile="wide", min_defects=1, max_defects=2, defect_size_range=(.30, .45),
                  crack_width_scale=1.0, crack_depth_scale=1.0, structures="off"),
    # Crazing is only visible on paint and stucco, so these walls draw one of those.
    "crazing": dict(crack_profile="crazing", min_defects=2, max_defects=3, defect_size_range=(.20, .45),
                    crack_width_scale=2.0, crack_depth_scale=2.0, crack_network_grime=(.8, 1.0),
                    crack_network_edge="organic", crack_network_extent=1.4,
                    structures="auto", structure_views=0),
}
# Close to the v1.9 training mix by view; major walls carry fewer cracks and so
# fewer views per wall.
UNIFIED_MIX = "standard=.72,wide=.17,crazing=.11"
# COMMON still carries --crack-path-mode angular; the later flag is the one argparse keeps.
RECIPES = {
    "unified": (COMMON.replace("--wall-materials diverse", "--wall-materials unified")
                + " --clean-wall-scale 0 --shell-finish matched --crack-style-mix " + UNIFIED_MIX
                + " --layout-order fixtures-first --fixture-mount relief"
                + " --crack-junction-fraction .5"
                + " --crack-path-mode angular_bounded"),
}


def parse_style_mix(text):
    """'standard=.72,wide=.17,...' -> {style: probability}, normalised."""
    mix = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        if name.strip() not in STYLES or not value:
            raise ValueError(f"crack style mix needs style=weight pairs from {sorted(STYLES)}: {text}")
        mix[name.strip()] = float(value)
    total = sum(mix.values())
    if total <= 0 or any(v < 0 for v in mix.values()):
        raise ValueError(f"crack style weights must be non-negative and not all zero: {text}")
    return {k: v / total for k, v in mix.items()}


def wall_style(seed, args):
    """(style name, a copy of args with that style applied) for one wall.

    The draw uses its own MD5-seeded stream, so it is the same in every process
    and leaves every other random stream alone. A network wall without a forced
    material draws a paint or stucco category from the same stream.
    """
    import hashlib
    import random
    from argparse import Namespace

    from syncrack3d.material_diversity import PAINTED_CATEGORIES, PAINTED_WEIGHTS

    rng = random.Random(int(hashlib.md5(f"{int(seed)}:crack_style".encode()).hexdigest()[:16], 16))
    mix = parse_style_mix(args.crack_style_mix)
    style = rng.choices(list(mix), weights=list(mix.values()), k=1)[0]
    wall = Namespace(**{**vars(args), **STYLES[style], "crack_style": style})
    if getattr(args, "layout_order", "legacy") == "fixtures-first" and wall.structures == "off":
        # Real rooms have doors and the other indoor fixtures whatever the crack;
        # the v1.x styles kept them to network walls only.
        wall.structures, wall.structure_views = "auto", 0
    if style == "crazing" and not args.material_category:
        wall.material_category = rng.choices(PAINTED_CATEGORIES, weights=PAINTED_WEIGHTS, k=1)[0]
    return style, wall


def recipe_argv(argv):
    """argv with the named recipe's flags placed before the user's own flags."""
    argv = list(argv)
    if "--recipe" not in argv:
        return argv
    i = argv.index("--recipe")
    if i + 1 >= len(argv) or argv[i + 1] not in RECIPES:
        raise SystemExit(f"--recipe needs one of {sorted(RECIPES)}")
    name = argv[i + 1]
    return RECIPES[name].split() + argv[:i] + argv[i + 2:] + ["--recipe", name]
