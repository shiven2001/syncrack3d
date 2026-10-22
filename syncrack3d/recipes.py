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

The labels and views of the released training set also passed a selection
step per batch; releases/*.json list the views each release kept.
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
