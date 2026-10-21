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

## v1.3.0 more views in the v1.2.0 mix, plus stone block and large tile cracks

The 1,912 v1.2.0 views are kept, and 169 views are added (138 with a crack,
31 without), for 2,081 views. 133 come from 30 walls in the v1.2.0 material
mix (seeds 176000–176029). 36 come from 8 cracked stone block and large tile
walls (seeds 176030–176037), the first cracks on those hosts. All use the
legacy profile and the angular path. Same 4,000-step, five-seed budget. The
recipe, seeds and the exact view list are in `releases/v1.3.0_added_views.json`.

Wood and irregular stone are now never cracked. Cracks on wood, stone block,
large tile and irregular stone as the whole 169-view addition cost CUBIT crack
F1 (−0.013) and BuildCrack F1 (−0.035). The same count of near-straight
cracks (`--crack-path-mode straight`) also lowered CUBIT F1 (−0.015) and BD3
recall. Both stay opt-in experiments, not part of v1.3.0.

BD3, mean of five seeds. Intervals resample duplicate groups and seeds.

| set | photos | model | crack IoU | centreline F1 | precision | recall | clDice |
|---|---:|---|---:|---:|---:|---:|---:|
| minor | 487 | v1.2.0 | 0.566 | 0.880 | 0.951 | 0.819 | 0.853 |
| minor | 487 | v1.3.0 | 0.573 | 0.883 | 0.961 | 0.817 | 0.857 |
| major | 358 | v1.2.0 | 0.443 | 0.839 | 0.962 | 0.745 | 0.814 |
| major | 358 | v1.3.0 | 0.438 | 0.832 | 0.970 | 0.729 | 0.807 |

Minor-crack IoU +0.006 [−0.010, +0.025] and precision +0.010 [+0.002, +0.021].
Plain-wall false alarms are 0.001.

CUBIT-InSeg at 1/3 scale. CR1 F1 +0.007 [+0.002, +0.012] and recall +0.015
[+0.007, +0.023]; all five seeds beat the v1.2.0 recall mean. Spalling photos
with a false crack are 49%.

| metric | CR1 v1.2.0 | CR1 v1.3.0 | all labelled v1.2.0 | all labelled v1.3.0 |
|---|---:|---:|---:|---:|
| centreline F1 at 3 px | 0.829 | 0.836 | 0.756 | 0.765 |
| precision | 0.918 | 0.913 | 0.758 | 0.760 |
| recall | 0.756 | 0.771 | 0.756 | 0.771 |
| clDice | 0.778 | 0.784 | 0.709 | 0.715 |
| crack IoU | 0.431 | 0.431 | 0.369 | 0.374 |

BuildCrack (355 positive tiles, source photos resampled): seed-mean F1 0.525
versus 0.516 for v1.2.0 (+0.009 [−0.053, +0.078]); empty-tile alarms fall
from 6.7% to 0%.

Held-out synthetic walls (6 unseen walls, seeds 176040–176045): stone block
crack recall 0.60 versus 0.50 for v1.2.0. Large tile was already 0.97. These
13 and 14 cracked views are a small test.

Without the stone block and large tile views, the same addition scores a
little higher on real photos (BD3 minor IoU 0.583, CUBIT CR1 F1 0.843) but
finds fewer stone block cracks (0.53). v1.3.0 keeps the stone block cracks.
Superseded by v1.4.0. BD3, CUBIT and BuildCrack have all been used to choose
between versions, so they are development scores.

## v1.4.0 crack networks (crazing)

The 2,081 v1.3.0 views are kept and 162 are added (131 with a crack, 31
without), for 2,243 views. They come from 30 paint and stucco walls (seeds
177000–177025 and 177030–177033) rendered with `--crack-profile network
--crack-width-scale 2 --crack-depth-scale 2 --crack-network-grime .8 1`. A
cracked view is kept only when its labelled lines are at least 3% darker than
the wall 2–4 px beside them; 13 of 153 were dropped. Same 4,000-step,
five-seed budget. The recipe, seeds and view list are in
`releases/v1.4.0_added_views.json`.

Why: on BuildCrack, v1.3.0 found 13% of the crack length on the 49 drone
tiles. Most of it is dense hairline crazing on painted render, a pattern no
earlier version rendered.

### Study

Each arm adds 131 cracked and 31 plain views to v1.3.0, from walls rendered
twice at the same seeds (same lights, cameras and defect positions). Five
seeds each. The choice rule was fixed before the last arm was scored: the
largest mean gain over v1.3.0 in BD3 minor IoU, CUBIT CR1 F1 and BuildCrack
F1; within 0.005, fewer BD3 plain-wall false alarms.

| arm | added cracks | BD3 minor IoU | CUBIT CR1 F1 | BuildCrack F1 | drone-tile F1 | BD3 plain alarms | CUBIT spalling false crack |
|---|---|---:|---:|---:|---:|---:|---:|
| v1.3.0 | – | 0.573 | 0.836 | 0.525 | 0.22 | 0.001 | 49% |
| control | v1.3.0 cracks, paint, stucco and concrete | 0.579 | 0.840 | 0.517 | 0.26 | 0.006 | 53% |
| faint networks | default staining (.15–.60), every wall | 0.592 | 0.843 | 0.590 | 0.57 | 0.020 | 69% |
| faint, half dose | faint networks on even seeds, control cracks on odd | 0.593 | 0.864 | 0.634 | 0.49 | 0.017 | 67% |
| **visible networks (v1.4.0)** | staining .8–1, ≥3% contrast, paint and stucco | **0.607** | 0.861 | 0.612 | **0.59** | 0.016 | 68% |
| visible, half dose | visible networks on even seeds, control cracks on odd | 0.597 | 0.862 | 0.602 | 0.49 | 0.007 | 58% |

Mean gains are 0.052 (faint, half dose), 0.049 (visible), 0.042 (visible,
half dose) and 0.030 (faint). The first two are within 0.005, and visible
networks have fewer plain-wall alarms. The control shows that 162 more
ordinary crack views change nothing, so the gain comes from the networks.

With default staining the networks are 2–4% darker than the wall at 640 px,
close to invisible. BuildCrack's drone crazing is about 4–8%.
`--crack-network-grime .8 1` gives 4–8% on paint and stucco and, against
faint networks, raises BD3 IoU by +0.015 [+0.007, +0.024] and CUBIT CR1 F1
by +0.019 [+0.015, +0.024]. On concrete the networks stayed invisible, so
v1.4.0 has none there.

Every network arm trades precision for recall. Raising the decision
threshold does not buy it back: at 0.95 the faint-network arm reaches CUBIT
precision 0.846 against v1.3.0's 0.913. Where false alarms cost more than
missed cracks, the visible half-dose arm has half the BD3 plain-wall alarms
of v1.4.0 for a smaller gain, and v1.3.0 has the fewest.

### v1.4.0 against v1.3.0

BD3, mean of five seeds. Intervals resample duplicate groups and seeds.

| set | photos | model | crack IoU | centreline F1 | precision | recall | clDice |
|---|---:|---|---:|---:|---:|---:|---:|
| minor | 487 | v1.3.0 | 0.573 | 0.883 | 0.961 | 0.817 | 0.857 |
| minor | 487 | v1.4.0 | 0.607 | 0.902 | 0.931 | 0.875 | 0.880 |
| major | 358 | v1.3.0 | 0.438 | 0.832 | 0.970 | 0.729 | 0.807 |
| major | 358 | v1.4.0 | 0.454 | 0.849 | 0.944 | 0.772 | 0.824 |

Minor IoU +0.034 [+0.024, +0.045], F1 +0.019 [+0.007, +0.031], precision
−0.030 [−0.044, −0.018]. Plain-wall false alarms rise from 0.001 to 0.016
(+0.015 [+0.004, +0.027]), about 6 of 355 photos. Some of those alarms
follow faint lines that may be unlabelled hairlines.

CUBIT-InSeg at 1/3 scale. CR1 F1 +0.025 [+0.011, +0.039]; all five seeds
beat the v1.3.0 mean. On all labelled photos F1 falls, −0.027 [−0.047,
−0.007], because spalling photos with a false crack rise from 49% to 68%.

| metric | CR1 v1.3.0 | CR1 v1.4.0 | all labelled v1.3.0 | all labelled v1.4.0 |
|---|---:|---:|---:|---:|
| centreline F1 at 3 px | 0.836 | 0.861 | 0.765 | 0.738 |
| precision | 0.913 | 0.839 | 0.760 | 0.634 |
| recall | 0.771 | 0.885 | 0.771 | 0.885 |
| clDice | 0.784 | 0.810 | 0.715 | 0.691 |
| crack IoU | 0.431 | 0.453 | 0.374 | 0.359 |

BuildCrack (355 positive tiles, source photos resampled): seed-mean F1 0.612
versus 0.525 (+0.088 [+0.030, +0.149]); 5-seed vote F1 +0.101 [+0.048,
+0.161]. Precision 0.784 versus 0.850, recall 0.504 versus 0.380. Drone tiles
F1 0.59 versus 0.22, other tiles 0.62 versus 0.58. Empty-tile alarms stay 0%.

Superseded by v1.5.0. It finds more cracks on all three benchmarks than
v1.3.0, at the cost of more false cracks on plain and spalled walls. All
three benchmarks have been used to choose between versions, so these are
development scores.

## v1.5.0 structures beside the v1.4.0 crack networks

The 2,081 v1.3.0 views stay. The same 162 v1.4.0 network views are
**replaced**, not added: the 30 paint and stucco walls are re-rendered from
the same seeds and cameras with `--structures auto` (105 indoor background
assemblies in total). Training stays at 2,243 views with the same crack
mix. Structures hide 6% of the crack pixels and fully cover the crack in 4
views, so the swap has 953 cracked views instead of 957. Exact stems:
`releases/v1.5.0_added_views.json`. Checkpoint:
`outputs/domain_transfer/syncrack3d_netstruct_bd3_v1`.

Confound: with structures on indoors the generator also closes the room
(side walls and a ceiling), which changes the lighting. A second control
re-renders the same walls in the same closed room with no structures
(`outputs/domain_transfer/syncrack3d_closedroom_bd3_v1`).

### Study

| | v1.4.0 | Closed room only | Structures swap | Swap − closed room [95%] |
|---|---:|---:|---:|---|
| BD3 minor-crack IoU | 0.607 | 0.606 | 0.605 | −0.001 (n.s.) |
| BD3 plain-wall false alarms | 1.6% | 2.1% | 2.0% | n.s. |
| CUBIT crack-photo F1 | 0.861 | 0.864 | 0.860 | −0.004 [−0.006, −0.001] |
| CUBIT spalling photos with false crack | 68% | 67% | 66% | |
| BuildCrack F1 | 0.612 | 0.628 | 0.613 | −0.015 (n.s.) |
| BuildCrack precision | 0.784 | 0.784 | 0.812 | +0.028 [−0.007, +0.065] |
| Crack-free walls with fixtures: views with a false alarm | 43% | 37.5% | 24% | |
| Crack-free walls with fixtures: false-crack area | 0.29% | 0.22% | 0.10% | |

n.s. = 95% interval includes zero. The last two rows are 56 held-out
crack-free synthetic views with doors, sockets, switches, vents, cables,
pipes, gutters, hatches and lights (`syncrack3d_structure_fp_probe_v1.json`);
none of those walls were used in training.

BD3, CUBIT and BuildCrack F1 stay within noise of v1.4.0. Swapping instead
of adding avoided diluting the crack networks. Against the closed-room
control the only significant change is a −0.004 dip in CUBIT F1. On the
clutter probe, comparing closed room alone to the swap isolates the
assemblies: false-alarm views fall from 37.5% to 24%, and false-crack area
falls about 2×. About a third of the gain versus v1.4.0 comes from the
closed room, which this design keeps because structures always come with
it. BuildCrack precision edges up (+0.028, borderline).

Caveats: the clutter test is small and synthetic, so it shows the model
learned to ignore these fixtures on generator walls, not that it will on
real photos. BD3 plain-wall false alarms and CUBIT spalling false cracks
are untouched. The real benchmarks barely contain clutter, apart from
BuildCrack's cable labels, so they cannot reward the structures.

v1.5.0 is the recommended zero-shot checkpoint: v1.4.0's crack accuracy
with fewer false alarms around fixtures, cables and frames. BD3, CUBIT and
BuildCrack remain development scores. v1.6.0 re-shades eight materials on
this same training list; the checkpoint path above is unchanged.

## v1.6.0 material shading

Same 2,243 views, same seeds. Eight materials were re-rendered (1,019
views): brick, cobble, concrete, large tile, marble, rubble, sandstone and
small tile. The other views are untouched. All 138 re-rendered walls matched
the originals on cameras, cracks, lights and room, so the shader edits kept
the random draws aligned. Cracked views per material stay the same except
cobble (39 to 38) and rubble (41 to 44).

Joints, dirt and grain sit on the existing Cycles displacement. Brick and
sandstone take a dirt amount from none to a light coat. Cobble drops the
round dome on each stone and puts soil in the joints. Concrete keeps its
chunks and adds a fine grain across the face. Large-tile grout is cement,
plaster or resin, and the glaze can be white, cream, gray or a colour.
Marble veins stay a little clearer. Rubble joints are soil, and the chip
waves are shading rather than extra displacement. On small tile the crack
used to replace the grout recess, so the joint was only a darker line; the
crack now adds onto a shallow cement joint.

A 5-seed retrain of that set, against v1.5.0:

| | v1.5.0 | New textures | Difference |
|---|---:|---:|---|
| BD3 minor-crack IoU | 0.605 | 0.602 | −0.003 (n.s.) |
| BD3 plain-wall false alarms | 2.0% | 1.5% | n.s. |
| CUBIT crack-photo F1 | 0.860 | 0.862 | +0.002 (n.s.); recall +0.007 ✓ |
| CUBIT spalling photos with false crack | 66% | 70% | |
| BuildCrack F1 (mean of seeds) | 0.613 | 0.630 | +0.018 (n.s.) |
| BuildCrack F1 (5-seed vote) | 0.626 | 0.647 | +0.022 ✓ |
| BuildCrack empty-tile false alarms | 0% | 13% | 2 of 15 checks, on 3 empty tiles |
| Crack-free walls with the new textures: views with a false alarm | 10.1% | 8.3% | |
| …on marble | 49% | 28% | |
| …on large tile / concrete | 3% / 8% | 9% / 11% | |

n.s. = 95% interval includes zero; ✓ = significant gain.

BD3 and CUBIT stay level. BuildCrack's 5-seed vote is a small significant
gain. Marble improves most: the old model false-alarmed on half the new
marble walls, because the sharper veins look like cracks, and training on
the new marble halves that. False cracks on CUBIT spalling photos rise from
66% to 70%, and false alarms on the new large tile and concrete rise
slightly. The empty-tile 13% is 2 alarms out of 15 checks. The last three
rows are 150 views of 16 unseen synthetic walls (2 per material) with the
new textures: a small synthetic check, not real-world evidence. No new
checkpoint is published; these scores are that 5-seed check.

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

# 4. score a crack model. Checkpoints are runs/s*/last.pt under --model.
#    BD3: 487 minor_crack + 355 plain (eval/bd3/README.md)
python eval/bd3/score.py --pred path/to/pred_dir
#    CUBIT-InSeg at 1/3 scale (1500x1000). Crack photos and spalling false alarms.
python eval/cubit/score.py --model path/to/checkpoint_dir --out path/to/cubit_out
python eval/cubit/compare.py --a path/to/cubit_a --b path/to/cubit_b --out compare.json
#    BuildCrack tiles, grouped by source photo. Pass one --model per checkpoint.
python eval/buildcrack/score.py --model v1.2.0=path/to/checkpoint_dir --out path/to/buildcrack_out
```

Defaults already match that crack recipe: width scale 1.3, and 70% of cracks
on paint, plaster and stucco use a coating edge. Brick, stone, concrete and
tile stay a single-material groove. The style
mix is 14% hairline, 16% hairline with a fork, 28% moderate, 20% branched, 16%
open (0.8–6 mm mouth) and 6% double run. `python syncrack3d/build_dataset.py --help`
lists every option.

| option | effect |
|---|---|
| `--material-category` | paint_neutral / paint_colour / paint_colour_mid / stucco_neutral / stucco_colour / brick / stone_block / irregular_stone / tile / large_tile / concrete / marble / wood (wood and irregular_stone are never cracked) |
| `--crack-layered-fraction` | coating edge on paint, plaster and stucco only. Paint: torn lips and 1.5–4 mm chips. Stucco: a broken finish coat, no lifted film. Brick, stone, concrete and tile are unchanged. Default 0.7 |
| `--crack-width-scale` | physical width multiplier, also written into the label. Default 1.3 |
| `--crack-render` | geometry (real displacement), pbr (bump only) or albedo (colour only) |
| `--crack-path-mode straight`, `--crack-axis-snap` | near-straight cracks; share turned to a wall axis (±2°). Opt-in ablation, not in v1.3.0 |
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

## Crack networks (v1.4.0)

Use `--crack-profile network` for map cracking/crazing: connected, irregular
closed cells and shared junctions across each defect patch. A smoothly warped
2-D Voronoi edge field drives both the shallow groove and its crack AOV.
This is a procedural appearance model, not a fracture simulation.

From this repository's root, with the usual Blender Python environment:

```bash
python -m syncrack3d.build_dataset --seeds 190010-190019 --output out/networks \
  --wall-materials diverse --material-category paint_neutral --clean-wall-scale 0 \
  --crack-profile network --crack-network-spacing .02 .06 \
  --crack-width-scale 2 --crack-label-mode support \
  --defect-size-range .3 .45 --min-defects 1 --max-defects 2 --max-props 0 \
  --resolution 1024 --samples 64 --dicing-rate 1 --wall-normal-mode explicit \
  --closeup-fill .9 1.1 --closeups-per-defect 2
```

For a mixture, keep `--crack-profile legacy` (or `major`/`mixed`) and add
`--crack-network-fraction .3`. That replaces approximately 30% of defect
instances, chosen deterministically from each instance seed. The default is
zero; existing profile/seed behavior stays unchanged.

`--crack-network-spacing MIN MAX` is the cell spacing in metres; smaller
values produce denser webs. Before width scaling, channel radii vary from
0.06–0.32 mm (full openings 0.12–0.64 mm), and groove depth from 0.04–0.18 mm.
Width/depth controls apply to networks. Coordinate warping slightly changes
local opening widths. Networks always use weak variable channel staining (including seeds in the
isolated-crack clean bucket) and omit the broad
coating chips and wall offsets used by large isolated cracks. Path fitting,
branch modes and path modes apply only to isolated cracks. Each network patch
is one crack instance; cells and edges are not separate instances. Manifests
record the sampled spacing, warp, seed offset, width and depth.

Resolution, standoff, exposure and lighting determine whether these hairlines
remain visible. Include close-ups and context views, and inspect export
curation: invisible/subpixel annotations can be ignored or held out rather
than accepted as training targets. The first 512 px, three-scene smoke test
produced six views with zero instance-contract issues, but only two close-ups
passed the existing evidence filter.

`--crack-network-grime MIN MAX` sets the channel staining as a share of crack
grime strength (default .15 .60). It only rescales the existing draw, so a
seed's cells, widths and depth stay the same. At 640 px the default leaves
lines 2–4% darker than the wall; v1.4.0 uses `.8 1` (4–8%) with width and
depth scale 2, and drops views below 3%. See the v1.4.0 study above.

Validation: 33 tests pass. The final controlled-wall pilot (seeds
190000–190002, 512 px) passes the label contract on six views, with two
close-ups retained. A 1024 px textured painted-wall check (seed 190010,
width scale 2, dicing rate 1) shows visible faint networks and passes the
contract on both views; its close-up is retained. A six-instance mixed
smoke scene (seed 190020, fraction .5) contains three crazing networks and
three ordinary cracks. These are generator checks, not transfer results.

## Structural background details (v1.5.0)

`--structures auto` adds an environment-dependent selection of background
assemblies: sagging cables with anchor hardware and slack loops, clipped
vertical cable bundles, conduit runs and elbows, downpipes with sockets and
brackets, utility boxes with lids/screws/feed conduit, framed windows with
mullions/sills/lintels, straight sealant panel joints, and projecting ledges.
Indoor auto sampling excludes the exterior downpipe/ledge assemblies.
v1.5.0 puts these beside the v1.4.0 crack networks by swapping the same
162 views; see the v1.5.0 study above.

These objects cast shadows and occlude the injected wall. They are **background**
in crack masks, including cable edges, joints and shadows. Hidden portions of
a crack lose their coverage naturally through Cycles visibility. Each assembly
has an independent seed, metric placement, object list and background semantic
record under `structures` in `scene.json`. They are never injection hosts.

Use a mixture of cracked and deliberately clean scenes so these details appear
without cracks as well as beside them:

```bash
python -m syncrack3d.build_dataset --seeds 192000-192019 --output out/structures \
  --environment mixed --structures auto --structure-views 2 \
  --structure-clean-fraction .35 --wall-materials diverse \
  --min-defects 1 --max-defects 3 --max-props 0 \
  --resolution 640 --samples 64 --wall-normal-mode explicit \
  --crack-label-mode support

python tools/prepare_defect_annotated_dataset.py -i out/structures/frames \
  -o out/structure_pack --aov-masks on --instance-masks on --with-bboxes \
  --max-negative-fraction .5
python tools/check_instance_contract.py --pack out/structure_pack
```

`--structures services` restricts sampling to cables, conduit, pipes and boxes;
`facade` samples windows, joints, ledges and downpipes. `all` attempts every
assembly for QA, skipping large pieces if no collision-free placement is found.
`off` is the default. Structural sampling uses its own random stream. Large
assemblies avoid crack centres and service footprints; thin services may cross
cracks. The existing `--max-props` controls furniture independently.

`--structure-views N` adds up to N views of distinct structural assemblies
(default 2), with role `structure` and no defect-target association. A structural
view can still contain a real crack, so its labels always come from the rendered
AOVs. `--structure-clean-fraction` defaults to zero; setting it to 1 generates
only clean scenes when structures are enabled. Existing material-specific clean
sampling still applies. Clean scenes now export explicit zero crack/instance
passes and an empty instance manifest. Keep scene grouping when splitting.

Window glazing is an opaque, recessed-looking surface within a raised frame,
not a boolean opening through the injection host. This provides window borders,
mullions and occlusion without labelling cracks behind glass. The assemblies
are procedural geometry approximations; they do not model entire building
interiors or construction systems. Panel joints are straight sealant strips,
not fracture fields. No real BuildCrack annotations are modified by these flags.

Validation on 2026-10-20: structure tests pass, including a rendered AOV
occlusion fixture. Three retained QA scenes cover indoor positives (seed
191000, 11 views), outdoor positives (191001, 10 views), and outdoor
negatives (191002, 12 views). All 33 views pass the instance/class/box
contract. All 12 clean crack AOVs and exported semantic masks are exactly
zero, and all 12 negatives survive curation. Structural views with no
visible crack are eligible negatives even when other views of the same
scene contain cracks. Views containing faint or under-resolution cracks
retain the existing evidence/ignore filtering. Transfer evidence for the
structures swap is in the v1.5.0 study above.
