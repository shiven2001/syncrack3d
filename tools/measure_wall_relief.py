"""Measure how far each Infinigen 2 wall category's relief stands out of the wall plane.

Bakes the material's displacement height (along the surface normal, metres;
syncrack3d/relief.py) onto a 2 m x 2 m patch at 2 mm texels for several seeds
per category, and prints the 99.9th percentile of outward height. The largest
per category is the fallback in ``RELIEF_M`` (material_diversity.py); builds
measure each wall's own material.

    python tools/measure_wall_relief.py [--seeds 8] [--categories a,b] [--per-seed]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RES = 1000  # 2 mm texels over the 2 m patch


def main() -> None:
    from syncrack3d.infinigen2_bridge import realize_material
    from syncrack3d.material_diversity import OVERLAY_FRACTION, RELIEF_M, WALL_CATEGORIES
    from syncrack3d.relief import bake_height

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--categories", default=",".join(WALL_CATEGORIES))
    parser.add_argument("--per-seed", action="store_true", help="also print each seed's p99.9 (mm)")
    args = parser.parse_args(argv)
    print(f"{'category':22s} {'p99.9 mm':>9s} {'max mm':>8s} {'min mm':>8s} {'current mm':>10s}")
    for category in args.categories.split(","):
        p999, peak, low = [], [], []
        for seed in range(1, args.seeds + 1):
            material = realize_material(category, seed, OVERLAY_FRACTION.get(category, 0.0))
            height = bake_height(material, res=RES)
            p999.append(np.percentile(height, 99.9))
            peak.append(height.max())
            low.append(height.min())
        print(f"{category:22s} {max(p999) * 1e3:9.1f} {max(peak) * 1e3:8.1f} {min(low) * 1e3:8.1f} "
              f"{RELIEF_M[category][0] * 1e3:10.1f}", flush=True)
        if args.per_seed:
            print("    per seed:", " ".join(f"{v * 1e3:.1f}" for v in p999), flush=True)


if __name__ == "__main__":
    main()
