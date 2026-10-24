"""Release recipes for build_dataset.py --recipe NAME.

Each recipe is the exact flag list the v1.8.0 training walls were rendered with.
build_dataset parses the recipe first and the command line after it, so any
flag given explicitly overrides the recipe. Material is not part of a recipe:
pass --material-category, or let --wall-materials diverse draw one per seed.

  legacy   indoor walls, legacy crack profile; some walls stay plain by material
           (the v1.0.0-v1.3.0 base walls; v1.3.0's added walls used
           --clean-wall-scale 0, every wall cracked)
  major    indoor walls, broad open and branched cracks, every wall cracked
  network  indoor paint and stucco walls with visible crack networks (crazing),
           organic outline, indoor structures beside the cracks (v1.7.0 on)
  facade   outdoor facades with facade structures, junction cracks and rain-wash
           staining (v1.8.0)
  unified  v2: one recipe for everything. Each wall draws a crack style
           (legacy/major/network/facade, STYLES) that sets only what defines that
           kind of crack; materials (--wall-materials unified: 15 crack-able
           categories drawn uniformly, no wood or irregular stone), capture and
           rendering are shared. Every wall is cracked (--clean-wall-scale 0),
           chosen after the 2026-10-23 clean-wall study: more BD3 recall at the
           cost of more false alarms than the material-aware table. Crack-free
           views still come from context views and plain close-ups.

The four release recipes rebuild the v1.x training walls; use unified for new
data.

The labels and views of the released training set also passed a selection
step per batch; releases/*.json list the views each release kept.
v1.0–v1.8 were rendered before v1.9.0 placed props by their real bounds;
add --prop-placement hint to rebuild their images exactly.
"""

COMMON = ("--substrate paint --max-props 1 --resolution 640 --samples 64 --wall-normal-mode explicit "
          "--dicing-rate .75 --exposure-ev-range .4 1.3 --closeup-fill .6 1.5 --closeup-aim-jitter .3 "
          "--preserve-closeup-scale --closeups-per-defect 2 --crack-fit-path --crack-label-mode support "
          "--crack-path-mode angular --crack-branch-mode local_curved --wall-materials diverse "
          "--awb-strength 0 .5 --crack-layered-fraction .7")
LEGACY = (COMMON + " --min-defects 2 --max-defects 3 --crack-width-scale 1.3 --crack-depth-scale .5 "
          "--defect-size-range .20 .45 --crack-profile legacy")

RECIPES = {
    "legacy": LEGACY + " --clean-wall-scale 1",
    "major": (COMMON + " --min-defects 1 --max-defects 2 --crack-width-scale 1 --crack-depth-scale 1 "
              "--defect-size-range .30 .45 --clean-wall-scale 0 --crack-profile major"),
    "network": (COMMON + " --clean-wall-scale 0 --min-defects 2 --max-defects 3 --defect-size-range .20 .45 "
                "--crack-profile network --crack-width-scale 2 --crack-depth-scale 2 --crack-network-grime .8 1 "
                "--structures auto --structure-views 0 --crack-network-edge organic --crack-network-extent 1.4"),
    "facade": (LEGACY + " --clean-wall-scale 0 --environment outdoor --structures facade --structure-views 0 "
               "--crack-junction-fraction .5 --crack-weathering-fraction .6"),
}

# v2: one recipe for everything. Each wall draws a crack style; the style sets
# only what defines that kind of crack (and its setting), everything else -
# capture, rendering, materials, clean-wall policy - is shared.
STYLES = {
    "legacy": dict(crack_profile="legacy", min_defects=2, max_defects=3, defect_size_range=(.20, .45),
                   crack_width_scale=1.3, crack_depth_scale=.5, environment="indoor", structures="off"),
    "major": dict(crack_profile="major", min_defects=1, max_defects=2, defect_size_range=(.30, .45),
                  crack_width_scale=1.0, crack_depth_scale=1.0, environment="indoor", structures="off"),
    # Crazing is only visible on paint and stucco, so these walls draw one of those.
    "network": dict(crack_profile="network", min_defects=2, max_defects=3, defect_size_range=(.20, .45),
                    crack_width_scale=2.0, crack_depth_scale=2.0, crack_network_grime=(.8, 1.0),
                    crack_network_edge="organic", crack_network_extent=1.4, environment="indoor",
                    structures="auto", structure_views=0),
    "facade": dict(crack_profile="legacy", min_defects=2, max_defects=3, defect_size_range=(.20, .45),
                   crack_width_scale=1.3, crack_depth_scale=.5, environment="outdoor", structures="facade",
                   structure_views=0, crack_junction_fraction=.5, crack_weathering_fraction=.6),
}
# Close to the v1.9 training mix by view (legacy ~75%, major ~10%, network ~7%,
# facade ~8%); major walls carry fewer cracks and so fewer views per wall.
UNIFIED_MIX = "legacy=.65,major=.15,network=.10,facade=.10"
RECIPES["unified"] = (COMMON.replace("--wall-materials diverse", "--wall-materials unified")
                      + " --clean-wall-scale 0 --crack-style-mix " + UNIFIED_MIX)


def parse_style_mix(text):
    """'legacy=.65,major=.15,...' -> {style: probability}, normalised."""
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

    from syncrack3d.material_diversity import PAINTED_CATEGORIES

    rng = random.Random(int(hashlib.md5(f"{int(seed)}:crack_style".encode()).hexdigest()[:16], 16))
    mix = parse_style_mix(args.crack_style_mix)
    style = rng.choices(list(mix), weights=list(mix.values()), k=1)[0]
    wall = Namespace(**{**vars(args), **STYLES[style], "crack_style": style})
    if style == "network" and not args.material_category:
        wall.material_category = rng.choice(PAINTED_CATEGORIES)
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
