"""Per-wall crack styles.

Every wall draws one style (``--crack-style-mix``, default ``DEFAULT_MIX``). A
style sets only what defines that kind of crack and its setting; materials,
fixtures, capture, lighting and rendering are shared by all walls.

  legacy   indoor walls, two or three thin to medium cracks
  major    indoor walls, one or two broad open and branched cracks
  network  indoor paint and stucco walls with crack networks (crazing)
  facade   outdoor facades with facade fixtures, cracks snapped to openings,
           sills and trims, and rain-wash staining

Indoor walls carry indoor fixtures (switched sockets, trims, doors, services)
whatever their style; ``--no-fixtures`` removes them for an ablation.
"""

STYLES = {
    "legacy": dict(crack_profile="legacy", min_defects=2, max_defects=3, defect_size_range=(.20, .45),
                   crack_width_scale=1.3, crack_depth_scale=.5, environment="indoor", structures="auto"),
    "major": dict(crack_profile="major", min_defects=1, max_defects=2, defect_size_range=(.30, .45),
                  crack_width_scale=1.0, crack_depth_scale=1.0, environment="indoor", structures="auto"),
    # Crazing is only visible on paint and stucco, so these walls draw one of those.
    "network": dict(crack_profile="network", min_defects=2, max_defects=3, defect_size_range=(.20, .45),
                    crack_width_scale=2.0, crack_depth_scale=2.0, crack_network_grime=(.8, 1.0),
                    crack_network_extent=1.4, environment="indoor", structures="auto"),
    "facade": dict(crack_profile="legacy", min_defects=2, max_defects=3, defect_size_range=(.20, .45),
                   crack_width_scale=1.3, crack_depth_scale=.5, environment="outdoor", structures="facade",
                   crack_junction_fraction=.5, crack_weathering_fraction=.6),
}
# Settings a style does not name.
STYLE_DEFAULTS = dict(crack_network_grime=(.15, .60), crack_network_extent=1.4, crack_junction_fraction=0.0,
                      crack_weathering_fraction=0.0)
# Close to the earlier training mix by view (legacy ~75%, major ~10%, network ~7%,
# facade ~8%); major walls carry fewer cracks and so fewer views per wall.
DEFAULT_MIX = "legacy=.65,major=.15,network=.10,facade=.10"


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
    and leaves every other random stream alone. A network wall draws a paint or
    stucco category from the same stream (within ``--material-categories``).
    """
    import hashlib
    import random
    from argparse import Namespace

    from syncrack3d.material_diversity import DEFAULT_CATEGORIES, PAINTED_CATEGORIES

    rng = random.Random(int(hashlib.md5(f"{int(seed)}:crack_style".encode()).hexdigest()[:16], 16))
    mix = parse_style_mix(args.crack_style_mix)
    style = rng.choices(list(mix), weights=list(mix.values()), k=1)[0]
    wall = Namespace(**{**vars(args), **STYLE_DEFAULTS, **STYLES[style], "crack_style": style,
                        "material_category": None})
    if args.no_fixtures:
        wall.structures = "off"
    if style == "network":
        allowed = args.material_categories
        if allowed is None:
            painted = [c for c in DEFAULT_CATEGORIES if c in PAINTED_CATEGORIES]
        else:
            painted = [c for c in PAINTED_CATEGORIES if c in allowed]
        if painted:
            wall.material_category = rng.choice(painted)
    return style, wall
