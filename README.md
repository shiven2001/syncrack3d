# SynCrack3D

Procedural cracks on painted, plastered and masonry walls, rendered in Blender
Cycles with pixel-exact crack labels, instance masks, boxes and COCO
annotations. This app generates cracks only. Paint peeling and blistering
are not part of SynCrack3D.

## v1.0.0 on BD3

Zero-shot scores for the crack model trained on 204 walls (17 materials × 12,
seeds 120000–120331). U-Net ResNet34, five seeds, final checkpoints. These
photos were inspected while the generator was designed, so the numbers are a
development score. Crack IoU counts only pixels predicted as crack. Centreline
F1 is pooled at a 3-pixel tolerance.

| set | photos | crack IoU | centreline F1 | precision | recall | clDice |
|---|---:|---:|---:|---:|---:|---:|
| minor cracks | 487 | 0.551 | 0.864 | 0.958 | 0.788 | 0.839 |
| major cracks | 358 | 0.339 | 0.765 | 0.969 | 0.633 | 0.741 |

On 355 plain walls the minor-crack model’s false-alarm rate is 0.002. A plain
photo counts as an alarm when predicted crack pixels exceed 0.1% of the image.
The BD3 photo files used for these scores are not included in this repository.

```
syncrack3d/      generator (wall, materials, cracks, cameras, lighting, render, ISP, manifest)
cwg_infinigen/   vendored subset of the Infinigen fork it builds on (materials, defect
                 injection/fields, AOV labels, sensor model, props) - see NOTICE.md
tools/           export (masks/splits/boxes/COCO) and label-contract check
isaac_port/      wall -> displaced, labelled USD panels in Isaac Sim, checked against Blender
```

## Setup

Python 3.11 with Blender-as-a-module:

```bash
pip install -r requirements.txt
```

## Generate, export, check

```bash
# 1. render walls (one seed = one wall, several close-up and context views)
python syncrack3d/build_dataset.py --seeds 107000-107011 --output out/walls \
  --material-category paint_neutral --awb-strength 0 .5 \
  --substrate paint --min-defects 2 --max-defects 3 --max-props 1 --resolution 640 --samples 64 \
  --wall-normal-mode explicit --dicing-rate .75 \
  --exposure-ev-range .4 1.3 --closeup-fill .6 1.5 --closeup-aim-jitter .3 --preserve-closeup-scale \
  --closeups-per-defect 2 --crack-fit-path --crack-label-mode support --crack-path-mode angular \
  --crack-branch-mode local_curved --crack-depth-scale .5 \
  --defect-size-range .20 .45 --wall-materials diverse --clean-wall-scale 0 --resume

# 2. export labels (0 background, 60 crack in masks/; scene-level train/val/test splits)
python tools/prepare_defect_annotated_dataset.py -i out/walls/frames -o out/pack \
  --aov-masks on --instance-masks on --with-bboxes --max-negative-fraction 1

# 3. check the instance/label contract
python tools/check_instance_contract.py --pack out/pack

# 4. score crack predictions on the frozen BD3 crack-only set
#    (487 minor_crack + 355 plain negatives; see bd3_eval/README.md)
python bd3_eval/score.py --pred path/to/pred_dir
```

Defaults already match that crack recipe: width scale 1.3, and 70% of cracks
use the two-layer edge. The style
mix is 14% hairline, 16% hairline with a fork, 28% moderate, 20% branched, 16%
open (0.8–6 mm mouth) and 6% double run. `python syncrack3d/build_dataset.py --help`
lists every option.

| option | effect |
|---|---|
| `--material-category` | paint_neutral / paint_colour / paint_colour_mid / stucco_neutral / stucco_colour / brick / stone_block / irregular_stone / tile / large_tile / concrete / marble / wood |
| `--crack-layered-fraction` | two-layer cracks: torn paint lips, 1.5–4 mm edge chips (labelled crack), plaster/cement/putty in the gap, 35% one-side offset. Default 0.7 |
| `--crack-width-scale` | physical width multiplier, also written into the label. Default 1.3 |
| `--crack-render` | geometry (real displacement), pbr (bump only) or albedo (colour only) |
| `--light-profile hard_mix`, `--relight-per-view` | sun / bare bulb / raking torch among the regimes; a new light rig per view |
| `--camera-isp`, `--awb-strength`, `--exposure-ev-range` | phone-camera processing, partial white balance, exposure spread |

Seeds are reproducible per wall: a wall's material, cracks, cameras and lights
depend only on its seed, not on which seeds ran before it in the same process.

## Isaac Sim port

Injected cracks exist only as shader displacement at Cycles render time, which
Isaac cannot evaluate, so a wall is ported per close-up view: the region each
close-up camera sees is baked to patch maps (true displaced relief measured with
an orthographic Position-pass render, colour, roughness, normal map, class
coverage) and the patches become displaced USD meshes with semantic labels.

```bash
python isaac_port/port_wall.py --seed 112000 --material-category paint_neutral --out out/port --blender-only
python isaac_port/port_wall.py --out out/port --isaac-only
```

## Maintainers

This folder is a snapshot of the repository generator, then edited here.
`controlled_wall_generator/` and `infinigen/` keep paint peeling and blistering,
including the two-layer peel. Rebuilding this app from the repo with
`python build_from_repo.py` would put that code back and overwrite the
crack-only edits; do that only after those edits have been ported, and pass
`--overwrite-local-edits` if you mean to discard them.
