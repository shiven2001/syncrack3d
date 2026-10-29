#!/usr/bin/env python3
# Copyright (C) 2025, SynCrack3D.

"""Optional snapshot of Infinigen 2 wall materials into a Blender library.

Runtime does not read this file. Wall seeds rebuild the vendored procfunc
shaders. This tool is only for inspecting a realized node tree.

Realizes each category, rejects any graph that lost Displacement, then writes
the datablock to ``syncrack3d/materials_v2/library.blend``.

    python tools/freeze_infinigen2_material.py
    python tools/freeze_infinigen2_material.py --categories paint_wall --seeds 0,3

Requires the optional bridge package (``requirements-infinigen2.txt``).
Runtime generators load from the .blend and do not import ``infinigen2``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "syncrack3d" / "materials_v2" / "library.blend"
MANIFEST = ROOT / "syncrack3d" / "materials_v2" / "library_manifest.json"


def _parse_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _displacement_ok(material: bpy.types.Material) -> dict:
    out = next(
        (n for n in material.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"),
        None,
    )
    if out is None:
        raise RuntimeError(f"{material.name}: missing Material Output")
    if not out.inputs["Surface"].is_linked:
        raise RuntimeError(f"{material.name}: Surface unlinked")
    if not out.inputs["Displacement"].is_linked:
        raise RuntimeError(
            f"{material.name}: Displacement unlinked — refuse to freeze a bump-only graph"
        )
    material.displacement_method = "DISPLACEMENT"
    disp_from = out.inputs["Displacement"].links[0].from_node
    return {
        "nodes": len(material.node_tree.nodes),
        "groups": sum(1 for n in material.node_tree.nodes if n.type == "GROUP"),
        "displacement_from": disp_from.bl_idname,
        "displacement_method": material.displacement_method,
    }


def freeze(categories: list[str], seeds: list[int]) -> dict:
    sys.path.insert(0, str(ROOT))
    from syncrack3d.infinigen2_bridge import INFINIGEN2_CATEGORIES, available, realize_material

    if not available():
        raise SystemExit(
            "infinigen==2.0.0a3 required: pip install -r requirements-infinigen2.txt"
        )
    unknown = sorted(set(categories) - set(INFINIGEN2_CATEGORIES))
    if unknown:
        raise SystemExit(f"unknown categories: {unknown}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict] = {}
    data_blocks = set()
    for category in categories:
        for seed in seeds:
            material = realize_material(category, seed)
            info = _displacement_ok(material)
            key = f"{category}@{seed}"
            entries[key] = {
                "category": category,
                "seed": seed,
                "material_name": material.name,
                "finish": INFINIGEN2_CATEGORIES[category],
                **info,
            }
            data_blocks.add(material)
            print(f"froze {key}: {info['nodes']} nodes, {info['groups']} groups")

    bpy.data.libraries.write(
        str(LIBRARY),
        data_blocks,
        fake_user=True,
        compress=True,
    )
    manifest = {
        "schema": "syncrack3d/materials_v2_library/1",
        "library": str(LIBRARY.relative_to(ROOT)),
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {LIBRARY}")
    print(f"wrote {MANIFEST}")
    return manifest


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from syncrack3d.infinigen2_bridge import INFINIGEN2_CATEGORIES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories",
        default=",".join(sorted(INFINIGEN2_CATEGORIES)),
        help="comma-separated category names",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(i) for i in range(8)),
        help="comma-separated integer seeds to freeze (runtime maps other seeds onto this pool)",
    )
    args = parser.parse_args()
    categories = _parse_list(args.categories)
    seeds = [int(s) for s in _parse_list(args.seeds)]
    freeze(categories, seeds)


if __name__ == "__main__":
    main()
