"""Default SynCrack3D scene options for porting a wall. Cracks only."""

COMMON = (
    "--substrate paint --min-defects 2 --max-defects 3 --max-props 0 --resolution 640 "
    "--samples 64 "
    "--wall-normal-mode explicit --dicing-rate .75 "
    "--exposure-ev-range .4 1.3 --closeup-fill .6 1.5 --closeup-aim-jitter .3 "
    "--preserve-closeup-scale --closeups-per-defect 1 --crack-fit-path --crack-label-mode support "
    "--crack-path-mode angular --crack-branch-mode local_curved --crack-width-scale 1.3 "
    "--crack-depth-scale .5 --defect-size-range .20 .45 --wall-materials diverse "
    "--clean-wall-scale 0 --crack-layered-fraction .7"
).split()


def common_options(a):
    return [*COMMON, "--material-category", a.material_category, *a.extra]
