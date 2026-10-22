"""--recipe: release flags come first, so explicit flags override them."""
import pytest

from syncrack3d.recipes import RECIPES, recipe_argv


def test_no_recipe_leaves_argv_alone():
    argv = ["--seeds", "1", "--output", "out"]
    assert recipe_argv(argv) == argv


def test_recipe_flags_precede_user_flags():
    argv = recipe_argv(["--seeds", "1", "--recipe", "facade", "--crack-weathering-fraction", "0"])
    assert argv[:len(RECIPES["facade"].split())] == RECIPES["facade"].split()
    # the user's value is the last occurrence of the flag, so argparse keeps it
    last = len(argv) - 1 - argv[::-1].index("--crack-weathering-fraction")
    assert argv[last + 1] == "0" and last > argv.index("--crack-weathering-fraction")
    assert argv[-2:] == ["--recipe", "facade"]


def test_every_recipe_uses_the_release_labels():
    for name, flags in RECIPES.items():
        assert "--crack-label-mode support" in flags, name
        assert "--dicing-rate .75" in flags, name


def test_unknown_or_missing_recipe_fails():
    for argv in (["--recipe", "nope"], ["--recipe"]):
        with pytest.raises(SystemExit):
            recipe_argv(argv)
