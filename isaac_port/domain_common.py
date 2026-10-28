"""Default SynCrack3D scene options for porting a wall. Cracks only."""

# Legacy-style cracks on an open room without fixtures or props, one close-up per crack.
COMMON = ("--crack-style-mix legacy=1 --no-fixtures --shell-finish plain --max-props 0 "
          "--closeups-per-defect 1").split()


def common_options(a):
    return [*COMMON, "--material-categories", a.material_category, *a.extra]
