"""The surrounding facade must not depend on Python's per-process string hashing."""
import os
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[2]
SNIPPET = (
    "from syncrack3d.outdoor import facade_extent, sample_outdoor_plan\n"
    "print([facade_extent(sample_outdoor_plan(s, 4.0)) for s in range(5)])\n"
)


def test_facade_extent_is_the_same_in_every_process():
    outs = set()
    for hash_seed in ("1", "2", "random"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run([sys.executable, "-c", SNIPPET], cwd=APP, env=env,
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
        outs.add(out)
    assert len(outs) == 1, outs
