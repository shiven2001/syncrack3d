"""Release recipes for build_dataset.py --recipe NAME.

Each recipe is the exact flag list the v1.8.0 training walls were rendered with.
build_dataset parses the recipe first and the command line after it, so any
flag given explicitly overrides the recipe. Material is not part of a recipe:
pass --material-category, or let --wall-materials diverse draw one per seed.

  standard indoor walls, the standard crack types; some walls stay plain by material
           (the v1.0.0-v1.3.0 base walls; v1.3.0's added walls used
           --clean-wall-scale 0, every wall cracked)
  wide     indoor walls, wide open and wide branched cracks, every wall cracked
  crazing  indoor paint and stucco walls with crazing,
           organic outline, indoor structures beside the cracks (v1.7.0 on)
  unified_v1.16  the unified recipe as of v1.16.0 (legacy layout order), to
           rebuild those walls exactly
  unified  v2: one recipe for everything. Each wall draws a crack style
           (standard/wide/crazing, STYLES) that sets only what defines that
           kind of crack; materials (--wall-materials unified: 10 crack-able
           categories drawn uniformly, one plaster and one stucco whose coats
           are neutral, coloured, or mid-tone, no wood or irregular stone), capture and
           rendering are shared. Every wall is cracked (--clean-wall-scale 0),
           chosen after the 2026-10-23 clean-wall study: more BD3 recall at the
           cost of more false alarms than the material-aware table. Crack-free
           views still come from context views and plain close-ups. Indoor side
           walls carry the test wall's finish and the floor an Infinigen floor
           material (--shell-finish matched, 2026-10-24 study). From v1.17 it
           uses --layout-order fixtures-first, and every indoor wall gets
           fixtures (closed room), whatever its crack style. Props always stand
           loose on the floor (the v1.18 rule). From v1.19 fixtures and the
           skirting stand on stone/brick relief instead of sinking into it
           (--fixture-mount relief). From v1.20 every fixture is placed by one
           rule (structures.sample_structure_plan): clear of every other.
           From v1.22 every wall is indoor: the facade style and the outdoor
           scene are gone, and a draw that would have been a facade wall
           draws again (REDRAW), so every other wall keeps its style.
           From v1.24 each crack token moves, with probability .5, to a corner
           or edge of a window, door, vent or access panel
           (--crack-junction-fraction .5, junctions.py).
  unified_v1.17, unified_v1.18, unified_v1.23  the unified recipe's flags as of
           that version (unified_v1.23: no junction cracks).
           Prop and fixture placement no longer differ between them.

The three release recipes keep the v1.x training walls' flags; use unified for
new data. The v1.8.0 facade recipe and its outdoor walls rebuild only from the
v1.21.0 tag.

The labels and views of the released training set also passed a selection
step per batch; releases/*.json list the views each release kept.
Props always use the loose rule and fixtures the v1.20 rule, so rebuilding a
wall from before v1.18 moves any prop in view, and a fixtures wall from before
v1.20 moves its fixtures; check out that version's tag to rebuild it exactly.
"""

COMMON = ("--substrate paint --max-props 1 --resolution 640 --samples 64 --wall-normal-mode explicit "
          "--dicing-rate .75 --exposure-ev-range .4 1.3 --closeup-fill .6 1.5 --closeup-aim-jitter .3 "
          "--preserve-closeup-scale --closeups-per-defect 2 --crack-fit-path --crack-label-mode support "
          "--crack-path-mode angular --crack-branch-mode local_curved --wall-materials diverse "
          "--awb-strength 0 .5 --crack-layered-fraction .7")
STANDARD = (COMMON + " --min-defects 2 --max-defects 3 --crack-width-scale 1.3 --crack-depth-scale .5 "
          "--defect-size-range .20 .45 --crack-profile standard")

RECIPES = {
    "standard": STANDARD + " --clean-wall-scale 1",
    "wide": (COMMON + " --min-defects 1 --max-defects 2 --crack-width-scale 1 --crack-depth-scale 1 "
              "--defect-size-range .30 .45 --clean-wall-scale 0 --crack-profile wide"),
    "crazing": (COMMON + " --clean-wall-scale 0 --min-defects 2 --max-defects 3 --defect-size-range .20 .45 "
                "--crack-profile crazing --crack-width-scale 2 --crack-depth-scale 2 --crack-network-grime .8 1 "
                "--structures auto --structure-views 0 --crack-network-edge organic --crack-network-extent 1.4"),
}

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
# A draw that lands on REDRAW draws again from the same stream. The unified mix
# keeps the retired facade style's 10% slot under this name, so every wall that
# was indoor before v1.22 keeps its style; the rest split .65:.15:.10.
REDRAW = "redraw"
# Close to the v1.9 training mix by view; major walls carry fewer cracks and so
# fewer views per wall.
UNIFIED_MIX = "standard=.65,wide=.15,crazing=.10,redraw=.10"
RECIPES["unified_v1.16"] = (COMMON.replace("--wall-materials diverse", "--wall-materials unified")
                            + " --clean-wall-scale 0 --shell-finish matched --crack-style-mix " + UNIFIED_MIX)
# v1.17: fixtures first, and fixtures on every indoor wall (wall_style below).
RECIPES["unified_v1.17"] = RECIPES["unified_v1.16"] + " --layout-order fixtures-first"
# v1.18 made loose placement the rule. It is now the only placement, so the
# recipe string no longer carries a flag for it.
RECIPES["unified_v1.18"] = RECIPES["unified_v1.17"]
# v1.19: fixtures mounted on the wall's relief. Keeping them clear of each other
# is now the only fixture rule, so the recipe no longer carries a flag for it.
RECIPES["unified_v1.23"] = RECIPES["unified_v1.18"] + " --fixture-mount relief"
# v1.24: half the crack tokens move to a corner or edge of a window, door, vent or
# access panel (junctions.py, its own stream), the setting of the v122jn screen.
RECIPES["unified"] = RECIPES["unified_v1.23"] + " --crack-junction-fraction .5"


def parse_style_mix(text):
    """'standard=.65,wide=.15,...' -> {style: probability}, normalised; may include REDRAW."""
    mix = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        if name.strip() not in (*STYLES, REDRAW) or not value:
            raise ValueError(f"crack style mix needs style=weight pairs from {sorted(STYLES)}: {text}")
        mix[name.strip()] = float(value)
    total = sum(mix.values())
    if total - mix.get(REDRAW, 0) <= 0 or any(v < 0 for v in mix.values()):
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
    style = REDRAW
    while style == REDRAW:
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
