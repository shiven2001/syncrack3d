# SynCrack3D

This package creates a simple, deterministic wall surface for defect-detection data. It is
deliberately separate from the Infinigen apartment solver. The base generator owns surfaces, materials,
architectural context, exact dimensions, seeds, camera/lighting setup, and manifests. It does **not**
currently own defects, render-engine settings, output settings, color management, or compositing.

## Current scope

The only implemented module is `flat`: one uninterrupted rectangular injection host. There are no
windows, doors, openings, corners, booleans, or bevels. Skirting is optional and disabled by default.

Implemented substrates are controlled painted plaster, concrete, and masonry. Their shaders contain
fine metric variation and bump only: no broad dirt, material-level ambient occlusion, booleans, bevel
modifiers, or true displacement. Defect injection can add true displacement later.

Every host:

- has a unique name ending in `.wall`;
- records `surface_finish` and `syndefect_surface`;
- has one unique Principled material with linked Base Color;
- is a manifold rectangular box with no modifiers;
- is listed in the scene manifest.

## Generate a base scene

From the repository root:

```bash
blender --factory-startup --background \
  --python syncrack3d/generate_scene.py -- \
  --output outputs/controlled_walls/seed_0001/scene.blend \
  --seed 1 --module flat --substrate paint
```

The generated scene includes one front-facing 50 mm camera, one area key light, and a Nishita Sky
world. Their deterministic values are recorded in the scene manifest. The command writes three files:

- `scene.blend` — geometry and materials only;
- `scene.spec.json` — the exact input specification, reusable with `--spec`;
- `scene.scene.json` — generated hosts, context objects, validation, and explicit exclusions.

Generate an exact preset:

```bash
blender --factory-startup --background \
  --python syncrack3d/generate_scene.py -- \
  --output outputs/controlled_walls/simple_painted_wall/scene.blend \
  --spec syncrack3d/presets/simple_painted_wall.json
```

Overall dimensions can also be fixed with `--width`, `--height`, and `--thickness`. Use
`--skirting` only when a skirting-board hard negative is specifically required.

## Defect-injection boundary

`CWG_DEFECT_TOKENS` is intentionally empty. Defect factories and placement remain outside this package
while their morphology is being developed.

When defect card objects are ready, bind each card to an exact host and call the adapter:

```python
from syncrack3d.injection_adapter import (
    DefectBinding,
    run_controlled_injection,
)

bindings = [
    DefectBinding(defect=crack_card, host=chosen_wall, surface="wall"),
]
run_controlled_injection(
    bindings,
    manifest_path="defect_instances.json",
    scene_id="controlled_0001",
    scene_seed=1,
)
```

The adapter creates the minimal explicit room relation expected by the existing
`cwg_infinigen.assets.defect_injection.run_injection`. It therefore reuses substrate checks, footprint
validation, host-material copying, displacement wiring, class/instance AOVs, shader-node budgets, and
the existing defect manifest. It never guesses the nearest wall.

## Intentional exclusions

This first stage does not configure:

- defect morphology or counts;
- defect placement;
- camera sampling or randomized viewpoints;
- Cycles/Eevee settings;
- compositor, exposure, sensor response, or output passes.

Those systems can be attached later without coupling them to base-scene generation.

## CWGv5 material diversity (opt-in, 2026-10-13)

`build_dataset.py --wall-materials diverse` (`material_diversity.py`) samples
each wall from a wide Infinigen palette instead of the near-neutral plaster
default. It targets the CWG4 real-photo failures: coloured/rough walls flooded
with false positives, and grout or mortar read as cracks.

| category | weight | generators | defect-free rate |
|---|---|---|---|
| paint_neutral | .22 | Plaster (default neutral paints) | .15 |
| paint_colour | .22 | Plaster, any hue, saturation .15-.51, value .35-.95 | .15 |
| concrete | .12 | Concrete | .35 |
| brick | .10 | Brick | .60 |
| tile | .14 | NonWoodTile | .60 |
| marble | .10 | Marble / MarbleRegular / MarbleVoronoi | .60 |
| wood | .10 | InteriorWood, plywoods, wood tiles | 1.0 (never cracked) |
| stone_block, large_tile | 0 (forced only) | StoneBlock, LargeTile | 1.0; cracked under `--clean-wall-scale 0` |
| irregular_stone | 0 (forced only) | IrregularStone | 1.0 (never cracked) |

- **Defect rules.** Peel and blister go on paint only. Cracks go on any
  substrate except wood and irregular stone, which stay plain even with
  `--clean-wall-scale 0`: cracks on them cost real-photo transfer in v1.3.0
  testing.
- **Defect-free walls** are close-up negatives: the camera frames virtual
  targets on plain wall.
- **Geometry fixes for patterned materials.**
  - Floor-designed patterns (tiles, wood tiles) are rotated onto the wall plane.
    Without this they rendered as endless vertical strips.
  - The wall gets metric UVs, without which bricks rendered about 1 m wide.
- **Options:**
  - `--material-category` forces one category (for QA);
  - `--clean-wall-scale` scales the defect-free rates; 0 cracks every wall
    except wood and irregular stone.
- **White balance.** Grey-world auto white balance drains a full-frame coloured
  wall to grey. Use `--awb-strength MIN MAX`, for example `0.3 0.7`, for
  partial correction per camera. The default is full correction, as before.
- **Infinigen change:** `plaster_paint_params` gained `hue_min`, `hue_max` and
  `sat_min`. Their defaults reproduce the old behaviour.
- **QA.** Run the CWG4 recipe plus `--wall-materials diverse --awb-strength 0.3 0.7`,
  with overviews
  `cwg5_material_overview.jpg`, `cwg5_clean_overview.jpg` and
  `cwg5_colour_check.jpg`.
  - 12 walls built, all 7 categories, 30 defect frames and 30 negatives.
  - Contract check: 0 issues, 0 blended pixels.
- **Not yet shown:** any effect on real transfer. Test it with a paired
  training comparison before adopting this as the CWGv5 default.
- **Export negatives correctly.** Use the exporter with `--aov-masks auto
  --instance-masks auto`. `on` rejects defect-free frames.

## Tests

The specification tests do not require Blender:

```bash
python3 -m pytest syncrack3d/tests
```

The Blender entry point validates that it generated exactly one active camera, one light, and one
Nishita Sky Texture, and that every host is modifier-free and injection-ready with a linked Base Color.

### CWGv5 sheet-peeling pilot

Use `--peel-style sheet --peel-substrate-palette sheet` for opt-in broad,
connected peeling with gypsum/concrete/primer/older-coat appearances. The
original chip-style default is preserved. The shared layout drives host
geometry, coverage labels and flap replay; it does not enlarge masks alone.
The sheet palette uses an explicit scene seed so repeated builds can vary
undercoat consistently by wall. The full pilot recipe, checks and transfer
results are recorded in [CWGv5_SHEET_PILOT.md](../CWGv5_SHEET_PILOT.md).
