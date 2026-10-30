#!/usr/bin/env python3
"""Generate a deterministic, defect-free controlled wall base scene.

This script must run in Blender or another Python environment that provides
``bpy``. It creates a front-facing camera, one area light, and a Nishita Sky
world, but no compositor, render configuration, or defect.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from syncrack3d.geometry import build_scene  # noqa: E402
from syncrack3d.manifest import (  # noqa: E402
    validate_base_scene,
    write_scene_manifest,
)
from syncrack3d.scene_setup import configure_capture_scene  # noqa: E402
from syncrack3d.spec import (  # noqa: E402
    MODULE_KINDS,
    SUBSTRATES,
    SceneSpec,
    sample_scene_spec,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output .blend path")
    parser.add_argument("--manifest", default="", help="Output JSON manifest path")
    parser.add_argument("--spec", default="", help="Use an exact previously saved SceneSpec JSON")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--module", choices=MODULE_KINDS, default=None)
    parser.add_argument("--substrate", choices=("random",) + SUBSTRATES, default="random")
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--thickness", type=float, default=None)
    parser.add_argument("--skirting", action=argparse.BooleanOptionalAction, default=None)
    if argv is None:
        script_args = (
            sys.argv[sys.argv.index("--") + 1 :]
            if "--" in sys.argv
            else sys.argv[1:]
        )
    else:
        script_args = argv
    return parser.parse_args(script_args or [])


def clear_scene():
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def generate(args):
    if args.spec:
        spec = SceneSpec.from_json(args.spec)
    else:
        spec = sample_scene_spec(
            args.seed,
            module=args.module,
            substrate=args.substrate,
            width_m=args.width,
            height_m=args.height,
            thickness_m=args.thickness,
            skirting=args.skirting,
        )
    random.seed(spec.seed)
    np.random.seed(spec.seed % (2**32))
    clear_scene()
    build = build_scene(spec)
    capture = configure_capture_scene(spec, build.root)
    validation = validate_base_scene(build, capture)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else output.with_suffix(".scene.json")
    )
    spec.to_json(output.with_suffix(".spec.json"))
    write_scene_manifest(manifest, spec, build, capture, validation)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    return output, manifest, validation


def main():
    output, manifest, validation = generate(parse_args())
    print(f"Controlled wall scene: {output}")
    print(f"Scene manifest: {manifest}")
    print(
        "Base-scene validation: "
        f"{validation['host_count']} hosts, {validation['context_count']} context objects, "
        "0 defects, 1 camera, 1 light, Nishita sky"
    )


if __name__ == "__main__":
    main()
