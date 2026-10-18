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
v1.1.0 left this checkpoint in place. v1.2.0 replaces it; see below.

## v1.1.0 major-crack profile

`--crack-profile major` and `mixed` are in this version. The default stays
`legacy`, so existing seeds keep their morphology. Geometry and labels share
the same crack field. A 52-wall major batch (13 materials × 4, seeds
140000–140123, 210 views) was added to the 1,702-view v1.0.0 set and trained
with the same 4,000-step, five-seed recipe.

BD3, mean of five seeds. Minor cracks did not improve. Major cracks did.

| set | photos | model | crack IoU | centreline F1 | precision | recall | clDice |
|---|---:|---|---:|---:|---:|---:|---:|
| minor | 487 | v1.0.0 | 0.551 | 0.864 | 0.958 | 0.788 | 0.839 |
| minor | 487 | v1.1.0 mix | 0.547 | 0.864 | 0.960 | 0.786 | 0.838 |
| major | 358 | v1.0.0 | 0.339 | 0.765 | 0.969 | 0.633 | 0.741 |
| major | 358 | v1.1.0 mix | 0.449 | 0.842 | 0.968 | 0.745 | 0.817 |

Plain-wall false alarms stayed about 0.001–0.002.

CUBIT-InSeg, zero-shot, 1/3 scale (1500×1000), mean of five seeds. The local
copy has no labelled test folder, so this is 126 CR1 crack photos and 272
labelled photos in total (146 of them spalling). Spalling polygons are ignored.
This is not an official CUBIT test split. The v1.1.0 mix scores slightly below
v1.0.0 because it predicts more false crack. Recall is about the same.

| metric | CR1 v1.0.0 | CR1 v1.1.0 | difference | all labelled v1.0.0 | all labelled v1.1.0 | difference |
|---|---:|---:|---|---:|---:|---|
| centreline F1 at 3 px | 0.818 | 0.807 | −0.011 [−0.017, −0.004] | 0.757 | 0.741 | −0.016 [−0.024, −0.007] |
| precision | 0.945 | 0.922 | −0.023 [−0.032, −0.015] | 0.797 | 0.766 | −0.031 [−0.045, −0.019] |
| recall | 0.723 | 0.719 | −0.004 | 0.723 | 0.719 | −0.004 |
| clDice | 0.771 | 0.758 | −0.012 | 0.711 | 0.695 | −0.017 |
| crack IoU | 0.434 | 0.418 | −0.016 | 0.381 | 0.361 | −0.020 |

At full resolution the CR1 F1 drop is larger: 0.767 versus 0.790. Only 2 of 5
mix seeds beat the v1.0.0 mean F1, and the photo-only 95% intervals leave out
seed variation, so the F1 gap is weak. The precision drop is consistent: all 5
mix seeds fall below the v1.0.0 mean. Spalling photos with a false crack rise
from 46% to 50%. The extra major-crack renders make the model fire more on
dark edges and on rough spalling or concrete. For a crack-only claim, the
v1.0.0 model was the better zero-shot checkpoint than this mix. The mix helps
wide cracks, which these CUBIT CR1 photos mostly do not contain.

## v1.2.0 host-gated coating edge

Paint and plaster still get the torn film: lips and 1.5–4 mm chips. Stucco
gets a thicker broken coat and no lifted lip. Brick, stone, concrete, tile
and marble stay a groove in the same material. The v1.1.0 training mix was
retrained after replacing those 577 views at the same seeds. Paint views were
left as released. Same 1,912 views and 4,000-step, five-seed budget.

BD3, mean of five seeds. Minor cracks improved. Major cracks stayed with the
v1.1.0 mix.

| set | photos | model | crack IoU | centreline F1 | precision | recall | clDice |
|---|---:|---|---:|---:|---:|---:|---:|
| minor | 487 | v1.0.0 | 0.551 | 0.864 | 0.958 | 0.788 | 0.839 |
| minor | 487 | v1.1.0 mix | 0.547 | 0.864 | 0.960 | 0.786 | 0.838 |
| minor | 487 | v1.2.0 | 0.566 | 0.880 | 0.951 | 0.819 | 0.853 |
| major | 358 | v1.0.0 | 0.339 | 0.765 | 0.969 | 0.633 | 0.741 |
| major | 358 | v1.1.0 mix | 0.449 | 0.842 | 0.968 | 0.745 | 0.817 |
| major | 358 | v1.2.0 | 0.443 | 0.839 | 0.962 | 0.745 | 0.814 |

Plain-wall false alarms are 0.002, the same rate as v1.0.0.

CUBIT-InSeg, same 1/3-scale setting as above. Crack F1 is above both earlier
checkpoints. The gain is recall. Precision is level with v1.1.0 and below
v1.0.0. Spalling photos with a false crack are 50%, matching v1.1.0 and above
v1.0.0’s 46%.

| metric | CR1 v1.0.0 | CR1 v1.1.0 | CR1 v1.2.0 | all labelled v1.2.0 |
|---|---:|---:|---:|---:|
| centreline F1 at 3 px | 0.818 | 0.807 | 0.829 | 0.756 |
| precision | 0.945 | 0.922 | 0.918 | 0.758 |
| recall | 0.723 | 0.719 | 0.756 | 0.756 |
| clDice | 0.771 | 0.758 | 0.778 | 0.709 |
| crack IoU | 0.434 | 0.418 | 0.431 | 0.369 |

The CR1 F1 difference versus v1.1.0 is +0.022 [+0.016, +0.028], and versus
v1.0.0 is +0.011 [+0.003, +0.019]. Four of five seeds beat each earlier mean.
The precision difference versus v1.0.0 is −0.027 [−0.039, −0.016]. At full
resolution CR1 F1 is 0.789, matching v1.0.0 (0.790) and above v1.1.0 (0.767).
v1.2.0 is the recommended zero-shot checkpoint: it keeps the major-crack gain
and improves minor cracks and CUBIT crack F1. The remaining gap versus v1.0.0
is CUBIT precision, not missed crack.

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
on paint, plaster and stucco use a coating edge. Brick, stone, concrete and
tile stay a single-material groove. The style
mix is 14% hairline, 16% hairline with a fork, 28% moderate, 20% branched, 16%
open (0.8–6 mm mouth) and 6% double run. `python syncrack3d/build_dataset.py --help`
lists every option.

| option | effect |
|---|---|
| `--material-category` | paint_neutral / paint_colour / paint_colour_mid / stucco_neutral / stucco_colour / brick / stone_block / irregular_stone / tile / large_tile / concrete / marble / wood |
| `--crack-layered-fraction` | coating edge on paint, plaster and stucco only. Paint: torn lips and 1.5–4 mm chips. Stucco: a broken finish coat, no lifted film. Brick, stone, concrete and tile are unchanged. Default 0.7 |
| `--crack-width-scale` | physical width multiplier, also written into the label. Default 1.3 |
| `--crack-render` | geometry (real displacement), pbr (bump only) or albedo (colour only) |
| `--light-profile hard_mix`, `--relight-per-view` | sun / bare bulb / raking torch among the regimes; a new light rig per view |
| `--camera-isp`, `--awb-strength`, `--exposure-ev-range` | phone-camera processing, partial white balance, exposure spread |

Seeds are reproducible per wall: a wall's material, cracks, cameras and lights
depend only on its seed, not on which seeds ran before it in the same process.

## Major cracks (experimental)

`--crack-profile legacy` preserves the original default and seed morphology.
`--crack-profile major` samples broad open and branched cracks equally;
`--crack-profile mixed` chooses major or legacy with probability 0.5 per defect.
These remain one **crack** class. No peeling/blister images are introduced.

Major cracks use a spatially varying channel radius of 1.5–3.5 mm at narrow
sections and 5–12 mm at wide sections, before `--crack-width-scale`; full
untapered openings are twice the radius (3–24 mm at scale 1). Tips narrow
further. Depth is 8–18 mm before `--crack-depth-scale`. The same field drives
geometry and coverage; existing layered paint-edge chips remain available.
These are procedural appearance ranges, not calibrated structural severity.
Historical `crack_width_*_m` manifest fields store radii; new
`full_opening_*_m` fields explicitly store full untapered opening bounds.

From this app's root, generate a separate candidate dataset:

```bash
python syncrack3d/build_dataset.py --seeds 130000-130002 --output out/major_qa \
  --crack-profile major --crack-width-scale 1 --min-defects 1 --max-defects 1 \
  --max-props 0 --resolution 640 --samples 64 --wall-normal-mode explicit \
  --dicing-rate .75 --closeup-fill .8 1 --preserve-closeup-scale \
  --crack-fit-path --crack-label-mode support --crack-path-mode angular \
  --crack-branch-mode local_curved --defect-size-range .3 .45 --save-blend
```

Use `mixed` for a candidate containing both fine and broad cracks; do not
resume into a legacy output directory (resume skips existing scenes).
The initial three-wall pilot used width scale 1.3, 384 px and 24 samples:
6 rendered/exported views, 4 retained by training curation; instance contract
0 issues and 0 blended pixels. Context views lose small cracks and the masks
contain disconnected pieces; this is a smoke test, not dataset acceptance.
19 tests pass; 1,000 legacy morphology seeds match the pre-change code.
The v1.1.0 scores above are from adding major-profile walls to the legacy set,
not from a matched legacy-versus-mixed retraining.
Centreline recall measures missing labelled centreline coverage; it does not
by itself show that insufficient predicted width explains the IoU gap.

### Can this simulate structural fracture?

Not yet. The current wall is a flat host with shader displacement, with no
loads, supports, elastic properties, fracture energy, reinforcement or masonry
interfaces. A wide procedural groove is not a structural simulation, and this
profile does not cover every building crack type.

A feasible separate extension is a quasi-static loaded-panel fracture solver
followed by a Blender importer. Start with a notched homogeneous panel under
tension/shear, specifying geometry, boundary conditions, elastic properties,
fracture energy and regularization length. Validate load–displacement response,
energy balance and mesh convergence before adding building-specific cases.
[FEniCSx/NewFrac](https://newfrac.github.io/fenicsx-fracture/notebooks/phase-field/phase-field.html)
and [MOOSE](https://mooseframework.inl.gov/source/materials/ComputeLinearElasticPFFractureStress.html)
provide coupled elasticity/fracture implementations suitable for evaluating
this route. Masonry needs explicit joint/interface behavior; reinforced
concrete needs an appropriate constitutive/reinforcement model.

Import crack geometry and displacement/opening estimates with units and solver
provenance. Do not use the diffuse phase-field damage-band width as the visible
crack opening or directly threshold it into a camera mask. Keep numerical
damage, exposed fracture geometry and rendered pixel coverage separate. Blender
then supplies materials, lighting, cameras and visible-surface annotations.
This solver integration has been assessed, not implemented or validated.

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
