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

v1.5.0 is the checkpoint for that structures swap: v1.4.0's crack accuracy
with fewer false alarms around fixtures, cables and frames. BD3, CUBIT and
BuildCrack remain development scores. v1.6.0 re-shades eight materials on
this same training list. v1.7.0 is the recommended checkpoint; see below.

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
checkpoint is published for the texture change alone; these scores are that
5-seed check. v1.7.0 is the recommended checkpoint.

## v1.7.0 organic networks

The v1.6.0 textures stay. The same 162 network views were re-rendered from
the same seeds and cameras with `--crack-network-edge organic
--crack-network-extent 1.4`, then trained for 5 seeds and compared with the
new-texture model whose networks are still square. Extent scales only the
network field. The crack cards and the cameras stay the same. At extent 1
an organic patch kept 57% of the square patch's crack length; at 1.4 it
keeps 95%. Checkpoint:
`outputs/domain_transfer/syncrack3d_netorgx_bd3_v1/`.

| | New textures, square networks | Organic ×1.4 | Difference |
|---|---:|---:|---|
| BD3 minor-crack IoU | 0.601 | 0.606 | +0.006 (n.s.) |
| BD3 plain-wall false alarms | 1.7% | 1.2% | n.s. |
| CUBIT crack-photo F1 | 0.862 | 0.863 | +0.001 (n.s.) |
| CUBIT spalling photos with false crack | 70% | 72% | |
| BuildCrack F1 (mean of seeds) | 0.630 | 0.633 | +0.002 (n.s.) |
| BuildCrack F1 (5-seed vote) | 0.647 | 0.649 | +0.001 (n.s.) |
| BuildCrack drone tiles F1 | 0.616 | 0.619 | |
| BuildCrack seed spread | 0.093 | 0.058 | |

n.s. = 95% interval includes zero.

The earlier organic drop was less crack, not the outline. With the crack
length matched, the two tie on every benchmark. Most rows lean slightly
toward organic, and none of the differences are significant. Organic is the
default for new data: no square edges, and no measurable cost. Square with
extent 1 remains the way to reproduce v1.4.0–v1.5.0.

Against v1.5.0, textures plus organic networks keep BD3 and CUBIT level and
raise BuildCrack's 5-seed vote from 0.626 to 0.649. The marble false-alarm
drop (49% to 28%) is the texture change from the v1.6.0 check.

## v1.8.0 facade junction cracks and rain-wash staining

The 38 walls v1.3.0 added (seeds 176000–176037, 169 views) are re-rendered as
outdoor facades with facade structures (windows, ledges, vents, hatches,
downpipes). Half of their cracks are moved to junctions with
`--crack-junction-fraction .5`: diagonal from opening corners, and along the
undersides and tops of sills and ledges. `--crack-weathering-fraction .6`
adds rain-wash dirt or efflorescence below 60% of cracks (the flag was removed
later, together with the outdoor wall weathering). Ordinary cracks
keep the default vertical orientation. The re-renders replace the indoor
versions, so the 2,243 view stems and everything else are as in v1.7.0.
50 cracks sit at junctions. Recipe and view list:
`releases/v1.8.0_replaced_views.json`. Checkpoint:
`outputs/domain_transfer/syncrack3d_facade2_bd3_v1/`.

Why: BuildCrack's misses concentrate on cracks at beam, ledge and opening
edges and on weathered facades, and SynCrack3D's cracks were never horizontal
(see `--crack-orientation`).

### Study

Four arms re-render the same 38 walls, five seeds each, protocol v2.

| arm | BD3 F1 | CUBIT CR1 F1 | BuildCrack F1 | BuildCrack vote F1 | BD3 plain alarms | CUBIT spalling false crack |
|---|---:|---:|---:|---:|---:|---:|
| v1.7.0 (indoor walls) | 0.909 | 0.875 | 0.696 | 0.710 | 1.6% | 80% |
| facade context only | 0.904 | 0.875 | 0.672 | 0.687 | 1.7% | 80% |
| + random orientation | 0.902 | 0.877 | 0.638 | 0.640 | 1.4% | 78% |
| + random orientation + junctions + staining | 0.912 | 0.878 | 0.704 | 0.727 | 2.1% | 82% |
| **junctions + staining, vertical cracks (v1.8.0)** | **0.910** | **0.878** | **0.709** | **0.723** | 2.0% | 80% |

Paired 5-seed-vote BuildCrack differences: facade context −0.023 [−0.037,
−0.010] against v1.7.0; random orientation a further −0.047 [−0.061, −0.036];
junctions and staining +0.087 [+0.064, +0.111] over random orientation.
Turned randomly with nothing around them, horizontal and diagonal cracks
teach the model that any line at those angles is a crack, and training already
rotates images by 90°. Horizontal cracks help where real ones sit: along
ledges, sills and opening corners. Without random orientation the result is
the same within noise (vote −0.004 [−0.013, +0.004]) with fewer false alarms.

### v1.8.0 against v1.7.0

| | v1.7.0 | v1.8.0 | difference [95%] |
|---|---:|---:|---|
| BD3 minor-crack F1 | 0.909 | 0.910 | +0.000 [−0.008, +0.008] |
| BD3 minor-crack IoU | 0.597 | 0.602 | +0.005 [−0.005, +0.015] |
| BD3 plain-wall false alarms | 1.6% | 2.0% | +0.4 [−0.7, +1.6] points |
| CUBIT CR1 F1 | 0.875 | 0.878 | +0.002 [−0.002, +0.006] |
| CUBIT CR1 recall | 0.889 | 0.896 | +0.007 [+0.000, +0.013] |
| CUBIT all-labelled F1 | 0.718 | 0.724 | +0.006 [+0.002, +0.011] |
| BuildCrack F1, mean of seeds | 0.696 | 0.709 | +0.013 [−0.013, +0.042] |
| BuildCrack F1, 5-seed vote | 0.710 | 0.723 | +0.012 [+0.004, +0.023] |
| BuildCrack non-drone tiles F1 | 0.701 | 0.716 | |

A small, consistent gain on the two building benchmarks and level on BD3,
with no significant rise in false alarms. All three benchmarks guided the
choice, so these are development scores. v1.8.0 remains the scored
checkpoint. v1.9.0 changes where props sit and was not retrained.

## v1.9.0 props placed by their real bounds

Props used to be dropped in from a catalogue footprint and then turned to a
random angle. Some objects also have their origin off-centre or at
mid-height, so the footprint was not the object that got rendered. On 40
indoor walls with up to 3 props each, 35 of 58 props overlapped something:
rugs went up to 2.5 m through the test wall and shelves 23 cm; props crossed
the side walls; props overlapped other props and wall fixtures; six food
boxes sat up to 13 cm below the floor; five props, mostly shelves, contained
a camera, so those views rendered the inside of a shelf.

Each prop is now measured after it is spawned and turned (`occlusion.py`,
`--prop-placement measured`, the default). It rests on the floor, with the
sampled distance from the wall as the real gap. It stays between the side
walls. It moves further out until it clears other props and fixtures;
furniture may still stand on a rug. Infinigen rugs are room-sized (3–7 m),
so they are shrunk to fit the floor. A prop that cannot fit, or that would
contain a camera, is dropped.

On the same 40 walls, 52 of 58 props remain and none clip. The only overlaps
left are props standing on rugs. Walls, cracks, cameras, lights and labels
are identical to before; only prop positions changed, and no new random
draws were added. The label check on re-rendered cracked walls is 14 frames,
14 boxes, 0 issues. 62 tests pass, including 8 in `test_props.py`.

The legacy, major and network recipes place one prop per wall, so rebuilding
a v1.0–v1.8 wall moves any prop that is in view. Add `--prop-placement hint`
to rebuild those images exactly as released; that reproduced all 40 walls.
No model was retrained, so there is no new checkpoint and no measured change
in benchmark scores. The scored checkpoint stays
`outputs/domain_transfer/syncrack3d_facade2_bd3_v1/`.

## v1.18.0 loose props

`--prop-placement loose` (in `--recipe unified`; needs `--layout-order
fixtures-first`) lets a prop stand anywhere on the floor in front of the
wall: its back 2 cm past the wall's displaced relief out to 0.9 m, turned any
way, redrawn up to 40 times. It must clear the wall, floor, ceiling, side
returns, fixtures, skirting, cameras (0.15 m), lights and other props, and may
hide at most half of any crack token face-on; otherwise it is dropped. Views
then jammed against a prop are dropped too. The catalogue and the one prop per
wall are unchanged, and nothing random is drawn after the props, so a wall
without a prop renders exactly as in v1.17.0. `--recipe unified_v1.17`
rebuilds v1.17.0.

Screen against v1.17.0 on the same 250 walls (seeds 910000-910249), U-Net R34,
4,000 steps, five seeds each, zero-shot, mean centreline F1: BD3 .875 -> .882
(seed sd .02), CUBIT .857 -> .858, BuildCrack .681 -> .679 (seed range up to
.12). Neutral within seed noise; kept because it looks like real rooms.

## v1.19.0 fixtures on the relief and clear of each other

`--recipe unified_v1.18` rebuilds v1.18.0.

`--fixture-mount relief` (in `--recipe unified`; fixtures-first only)
stands every fixture and the skirting out from the wall plane by the wall's
own displaced relief, so stone and brick no longer poke through them; before
this a cable or conduit on a cobble wall was half-buried in the stones. The
window sill still reaches 1 cm into the wall, as built. Screen against loose
props alone (v1.18.0) on the same 250 walls and protocol: BD3 .882 -> .873 (seed sd
.03), CUBIT .858 -> .865, BuildCrack .679 -> .666 (seed range up to .12).
Neutral within seed noise; kept because it removes a visible clipping
artefact.

`--fixture-spacing clear` (also in `--recipe unified`; replaced in v1.20.0 by
the one fixture rule) keeps fixtures apart.
Before it, only large assemblies avoided each other: thin runs were placed
blind, and 54 of the 250 screen walls had a clash such as trunking through a
window, plumbing or conduit under a sill, a cable across trunking, or a
conduit through the skirting. Now no fixture comes within 3 cm of another's
built extent (`built_extent`: a window's sill and lintel, a conduit's elbow,
a cable's sag and loop; a test checks it covers the built geometry), vertical
runs stop at the skirting and ceiling trim, and a fixture with no room is
dropped. Panel seams are flush joints and stay exempt. 176 of the 250 walls
keep their plan exactly. Screen against the relief mount on the same walls and
protocol: BD3 .873 -> .877 (seed sd .01), CUBIT .865 -> .863, BuildCrack .666
-> .647. Neutral within seed noise, though BuildCrack has drifted .679 -> .647
over the two fixture changes; a 1,000-wall check before release settles it.

## v1.20.0 one fixture rule

Every fixture is now placed by one rule (`structures.sample_structure_plan`),
the only fixture placement; `--fixture-spacing` is gone. Each kind differs
only by its row in a table: a size, an anchor for its centre height (anywhere,
a band such as 0.25-0.45 m for a socket, a run from the skirting to under the
ceiling trim, standing on the floor, or a full-width trim) and the hardware it
builds past its nominal box (`PAD`: a window's sill and lintel, a utility
box's feed, a cable's sag). A fixture's built extent must stay on the wall and
3 cm clear of every fixture already placed and of the base skirting; a large
kind also keeps crack centres 6 cm clear. One that finds no room in 60 draws
is dropped. A door and the trims go first and may meet each other; panel
seams are flush joints that others may cross. The door-avoidance and
large-assembly checks with their own gaps and nominal sizes, and the special
height for a cable, are gone. The gutter is 12 mm narrower, so its end caps
no longer reach into the side walls.

v1.19.0's clear spacing only kept fixtures off each other. On 2,000 auto
walls per environment it still put hardware past the wall edge on 139 indoor
and 184 outdoor walls: a utility box's feed or a window lintel into the
ceiling, a sill past a side wall or into the floor. The one rule has none,
keeps as many fixtures (3.00 against 3.00 indoor, 2.89 against 2.85 outdoor)
and plans in the same 0.02 ms.

Screen against v1.19.0 on the same 250 walls and protocol, with only the
planner swapped into the v1.19.0 tag so props match (every wall's fixture
plan changed; 760 fixtures against 758): BD3 .877 -> .883 (seed sd .01),
CUBIT .863 -> .865, BuildCrack .647 -> .657 (seed range .13), BD3 plain-wall
false alarms .035 -> .038. Neutral within seed noise; kept because it is
simpler and nothing clips. Rebuilding a fixtures wall from before v1.20.0
moves its fixtures; check out that version's tag to rebuild it exactly.
Script: `studies/syncrack3d_fixture_rule_250.sh`.

## v1.21.0 one fixture catalogue, and the outdoor screen

The fixture catalogue is smaller and every service run is full length. Sockets,
switches, the sampled skirting and ceiling trims, the ledge and the outdoor-only
kinds (exterior vent, service hatch, gutter, wall light) are gone. Door and
trunking are indoor only; vent, access panel and the wire, conduit, downpipe
and plumbing runs are shared. A wire, conduit, pipe, plumbing run, trunking or
panel seam now spans the wall's clear height or its full width (each draw picks
one), with its own hardware: three insulated cables in clips, PVC or EMT
conduit with couplers and saddles, a PVC rainwater pipe with sockets and
clips, and a painted utility cabinet with hinges, latch and sometimes a meter.
Outdoors the built run continues across the whole facade; the plan stays on
the inspection wall, so cracks and junctions do too. Matched side walls now
always carry the test wall's own finish (bump only), and the unified palette
leaves out rubble stone (10 categories), which at room scale reads as the same
irregular stone wall as cobble. None of these changes has its own screen yet.

Outdoor screen, run before those changes on v1.20.0 with combined plaster and
stucco and no outdoor dirt (v120nd): its 22 outdoor (facade) walls replaced by
the next 22 indoor walls (910250 onward), so both train on 250 walls; same
protocol. BD3 .864 -> .875 (seed sd .022 / .012; higher on all five seeds),
plain-wall false alarms .055 -> .046, CUBIT .860 -> .859, BuildCrack .666 ->
.683 (seed sd .03-.04). The outdoor walls neither help nor hurt on these three
test sets, none of which is mainly outdoor walls. The replacements include 4
major-crack and 1 network wall where the facade walls were all legacy cracks.
Script: `studies/syncrack3d_indoor_only_250.sh`.

## v1.25.0 Isaac port for the indoor generator

The generator is unchanged; the Isaac Sim port is brought up to date and
checked end to end on Isaac Sim 5.0.0: a v1.24.0 stone-block room (seed
910004: export, compose, previews, embodied views) and close-up panels on a
painted wall (seed 112000) and a stone-block wall (seed 112001).

- Panel relief is measured with the inspection wall rendered alone, and every
  sample whose ray hits it counts. The old -6..+8 mm band cut masonry joints
  (down to -10.6 mm on stone block) out of the panel mesh. Baked against
  rendered relief: correlation .97-.996, slope .95-1.02.
- The Blender light references switch off every world background node. The
  sky node is `CWG_Sky_Background`, so the old lookup of `Background` left it
  on and the sky lit the wall evenly through the room's open side: the
  crack's response to the side of the raking light was .009 in Blender
  against .11-.36 in Isaac. It is now .11-.19.
- `compose_room.py` drops its outdoor branches (every scene is indoor from
  v1.22) and labels the skirting a fixture.

## v1.24.0 junction cracks by default

`--recipe unified` now moves cracks to openings
(`--crack-junction-fraction .5`): each crack token, with probability .5,
moves to a corner of a window, door, vent or access panel and runs 30-60°
off horizontal away from it, or lies along a door jamb or under a window sill
on a near-straight path (`syncrack3d/junctions.py`, its own random stream).
Walls without such an opening, and tokens not drawn, are unchanged. Up to
v1.23 the step was used only on facade walls. `--recipe unified_v1.23` keeps
the old flags.

A moved crack now also keeps the plan's own spacing from every other crack
(0.30 m, and the full patch extent for network cracks), not just 0.4 × their
summed sizes. Without it, 2 of 30 network walls in the screen below had
patches 0.35 and 0.54 m apart against the 0.63 m that keeps their instance
ids from blending, and 3 of 128 other walls had cracks closer than 0.30 m.

Screen on the v1.22.0 code, seeds 910000-910249, same protocol as v1.22.0
(250 walls, U-Net R34, 4,000 steps, five seeds, zero-shot, 5 px): 143 walls
carry 203 junction cracks (155 corner, 48 edge).

| | v1.22.0 | + junctions .5 |
|---|---|---|
| BD3 F1 | 0.887 (sd .014) | **0.898** (sd .016) |
| BD3 precision / recall | .944 / .836 | .940 / **.860** |
| BD3 plain-wall false alarms | **.039** | .082 |
| BD3 crack IoU | .581 | **.599** |
| CUBIT F1 | .863 | .866 |
| BuildCrack F1 | .640 (sd .037) | **.732** (sd .016) |

BuildCrack gains on every seed (.713-.757 against .593-.695), from recall
(.563 -> .711). The BD3 gain is within seed noise; false alarms on plain BD3
walls double, on every seed. The v1.23.0 fixtures and this spacing rule have
no screen of their own yet. Script: `studies/syncrack3d_v122_junctions_250.sh`.

## v1.23.0 fixture surfaces

Doors are raised panels with a ranged finish and fine grain. Access panels
have no hinges. Vents, panel seams, and the service runs (wires, conduit,
pipe, plumbing, trunking) draw a colour and a texture; those runs draw 2, 3
or 4 in parallel. The loose cable is in the indoor pool and spans the wall.
Utility cabinets are enclosed boxes: a hinged door, a screwed cover, a meter,
or louvres, with a ranged paint or galvanized finish and a top feed.

Window sills and heads use the inspection wall's own material, with real
displacement. The pane is a pale mirror, so it reads as glass without lighting
the frame; a dark backing still hides the host. Junction cracks sit on the
drawn sill and head.

Fixture surfaces use a fine bump. Props that already output a height stay
bump, and a flat prop material gets a fine bump of its own. The inspection
wall is still the only surface with real displacement, together with the
window sill and head that share it.

## v1.22.0 indoor only

SynCrack3D is now an indoor generator. The outdoor scene is removed:
`syncrack3d/outdoor.py` (ground, sky, sun, surrounding facade, corner, eave,
downpipe), the `--environment` flag, the `facade` crack style, the `facade`
release recipe, the `facade` and `outdoor` fixture profiles, and the facade
spans of service runs. `scene.json` still records `"environment": "indoor"`
so manifests read like the v1.x ones; `room_shell` is always set and the
`outdoor` record is gone. The synthetic evaluation set loses its facade
subset, and its negatives are indoor only.

The unified mix keeps the facade style's 10% slot as `redraw`: a wall that
lands on it draws its style again from the same stream. Every wall that was
indoor in v1.21.0 keeps its style and settings (checked on seeds
910000-919999: 9,014 unchanged); the 986 former facade seeds become 733
legacy, 140 major and 113 network walls. Outdoor walls rebuild only from the
`v1.21.0` tag.

Screen (250 walls, seeds 910000-910249, every wall rendered with this code;
same protocol as v1.21.0): BD3 F1 .887 (sd .014), precision .944, recall
.836, plain-wall false alarms .039, crack IoU .581; CUBIT .863; BuildCrack
.640 (sd .037). Against v1.20.0 with indoor walls only (.875 / .046 / .859 /
.683) BD3 is higher on 4 of 5 seeds and BuildCrack lower on 4 of 5, both
about one seed sd. It includes the v1.21.0 fixture, side-wall and palette
changes as well as the redrawn facade seeds. Script:
`studies/syncrack3d_v122_250.sh`.

## v1.10.0 whole rooms for embodied inspection

Close-up training cameras are aimed at the cracks, so a policy trained on
them never has to search. This version exports a composed room to Isaac Sim 5
(`isaac_port/export_room.py`, `isaac_port/compose_room.py`) and measures the
v1.8.0 detector on mobile views of that room (`render_room_views.py`,
`room_false_alarms.py`). The bake and the 3-room results are in the Isaac Sim
section below.

On 3 rooms × 60 path-traced poses, 42% of views have a false alarm under the
room lamps and 37% under hard lamps. Object edges are 84% of those
false-alarm pixels. Long cast-shadow edges rarely fire. Visible cracks found:
31% overall, and all 4 cracks larger than 600 px. These are development
measurements of the v1.8.0 checkpoint, not a new trained model.

`eval/synth/build.py` builds a held-out synthetic set from seeds 800000
upward, one subset per release recipe plus crack-free negatives.
`eval/synth/score.py` scores a checkpoint on it. No scores from that set are
reported here. The scored crack checkpoint stays
`outputs/domain_transfer/syncrack3d_facade2_bd3_v1/`.

```
syncrack3d/      generator (wall, materials, cracks, cameras, lighting, render, ISP, manifest)
cwg_infinigen/   vendored subset of the Infinigen fork it builds on (materials, defect
                 injection/fields, AOV labels, sensor model, props) - see NOTICE.md
tools/           export (masks/splits/boxes/COCO) and label-contract check
isaac_port/      wall -> displaced, labelled USD panels in Isaac Sim, checked against Blender
```

## Setup

Two Python environments: one renders, one scores.

```bash
# generator: Python 3.11 (bpy 4.2 needs it); Blender runs as a Python module
pip install -r requirements.txt
python -m pytest syncrack3d/tests          # 79 tests, a few seconds

# scoring (any Python with CUDA PyTorch)
pip install -r eval/requirements.txt
```

Cycles renders on the first GPU it finds (OptiX, CUDA, HIP, Metal, oneAPI)
and otherwise on the CPU. Blender prints `Error: Not freed memory blocks`
when a run exits; it is harmless. Run commands from this folder.

## Generate new data: the unified recipe (v1.24.0)

For new data use one recipe for everything. The commits that built it are
tagged:

| version | commit | what it adds |
|---|---|---|
| v1.11.0 | `76bfbd4` | `--recipe unified`: legacy 65%, major 15%, network 10%, facade 10%; 15 crackable materials drawn uniformly |
| v1.12.0 | `039d2c8` | `--clean-wall-uniform`, one clean-wall probability for every material |
| v1.13.0 | `f95dadc` | clean-wall study: material-aware beats all-cracked and uniform |
| v1.14.0 | `3b2d31e` | every wall cracked; matched indoor finishes, opt-in; per-wall timing |
| v1.15.0 | `089fbf2` | matched-finish study |
| v1.16.0 | `9617872` | matched finishes are the unified default |
| v1.17.0 | `85a666c` | fixtures first, on every indoor wall (`--recipe unified_v1.17`) |
| v1.18.0 | `4c698ea` | loose props: anywhere on the floor 2 cm past the wall relief to 0.9 m out, any turn, hiding at most half of a crack (`--prop-placement loose`; `--recipe unified_v1.18`) |
| v1.19.0 | tag `v1.19.0` | fixtures and skirting stand on the wall relief (`--fixture-mount relief`) and clear of each other (`--fixture-spacing clear`) |
| v1.20.0 | tag `v1.20.0` | every fixture placed by one rule: size, anchor and hardware per kind; on the wall and 3 cm clear of every other |
| v1.21.0 | tag `v1.21.0` | one smaller fixture catalogue with full-length service runs; matched side walls always; rubble stone out of the unified palette |
| v1.22.0 | tag `v1.22.0` | indoor only: the outdoor scene, facade style and recipe are removed; facade draws redraw |
| v1.23.0 | tag `v1.23.0` | ranged indoor fixtures on a fine bump; window sill and head use the inspection wall, displaced (`--recipe unified_v1.23`) |
| v1.24.0 | tag `v1.24.0` | half the cracks moved to window, door, vent and access-panel corners and edges (`--crack-junction-fraction .5`); moved cracks keep the plan's spacing |
| v1.25.0 | tag `v1.25.0` | Isaac port: wall-only panel relief without a band, isolated Blender light references, indoor-only room compose (generator unchanged) |

The three release recipes stay on the plain shell, so the v1.0–v1.10 walls
rebuild as released.

```bash
python syncrack3d/build_dataset.py --recipe unified --seeds 0-99 --output out/walls --resume
```

Each wall draws a crack style:

| style | probability | walls |
|---|---|---|
| legacy | 0.65 | indoor, legacy cracks |
| major | 0.15 | indoor, broad open and branched cracks |
| network | 0.10 | indoor paint or stucco, crack networks beside indoor fixtures |
| redraw | 0.10 | the retired facade style's slot: the wall draws again (v1.22.0) |

Every wall is indoor (v1.22.0). The redraw slot keeps every wall that was
indoor before the same, so the split is .72 / .17 / .11 in practice. The style
sets only what defines that kind of crack. Everything else is shared.

- **Materials:** `--wall-materials unified` draws 10 categories uniformly
  (one plaster and one stucco, each with a neutral, coloured or mid coat).
  The v1.x training data was built in near-equal forced batches per category,
  so this matches what the model was trained on, not the old palette weights.
  Wood, irregular stone and rubble stone are left out.
- **Clean walls:** none. Every wall is cracked (`--clean-wall-scale 0`),
  chosen after the clean-wall study below. The material-aware table
  (`--clean-wall-scale 1` with `--wall-materials unified`: paint and stucco
  0.15, concrete 0.35, patterned masonry, tile and stone 0.60) stays
  available.
- **Indoor context:** `--shell-finish matched`. Side walls carry the test
  wall's finish (bump only; from v1.21.0 in every room, before that in 70%),
  and the floor gets an Infinigen floor material. See the matched indoor
  finishes study below.
- **Shared settings:** capture, lighting, rendering and labels.

`--crack-style-mix` changes the mix (for example `network=1` for one style),
and `off` disables per-wall styles. `scene.json` records each wall's
`crack_style`.

The unified recipe has not been trained on yet. Adopting it needs the usual
paired comparison against v1.9.0 on BD3, CUBIT and BuildCrack.

### Layout order: fixtures first (opt-in, 2026-10-25)

`--layout-order fixtures-first` builds a wall in this order:

1. wall and material
2. room shell
3. fixtures, planned and built
4. crack tokens, then junction snapping
5. crack injection
6. lighting and cameras
7. props

The default, `legacy`, keeps the v1.x order:

1. crack tokens
2. props
3. lighting
4. fixture plan, conditioned on the cracks
5. junction snapping
6. injection
7. fixture meshes
8. cameras

The legacy order existed so that adding fixtures in v1.5 moved no crack or
camera, but it is backwards physically: cracks start at openings, not the
other way round. Checks in the new order:

| step | check |
|---|---|
| fixtures | planned on their own stream, blind to cracks; footprints reserved between fixtures as before |
| crack tokens | no centre under a large assembly (windows with sill and lintel, doors, boxes, vents, sockets, switches, lights), grown by 0.3 × the largest token size + 3 cm; 30% of tokens are tried 3–15 cm beside a cable, pipe or trunking run (`--crack-near-service-fraction`), so services partly cover some cracks; network tokens are spaced by their full patch size (extent × size), because overlapping patches blended instance ids |
| junctions | a share of tokens (.5 in the unified recipe from v1.24) moves to opening corners and edges, keeping the plan's spacing from every other token (network: full patch extent) |
| cameras | a ray along each view axis, ignoring the wall and crack cards: a camera with another object within 0.35 m and before half the way to the wall is dropped |
| props | placed around the cameras (0.2 m boxes) and the fixture meshes, stepping out from the wall past them or dropped |

QA (12 walls with fixtures, seeds 840000–840011, network and facade styles):

- every requested crack was placed, and none has its centre under a large
  fixture;
- junction cracks still occur on facades;
- the label contract was clean (79 frames, 0 issues);
- two cameras were dropped, both real faults: a facade close-up 5 cm behind
  the downpipe (the near-black views of v1.8.0), and an indoor context camera
  inside a side wall.

The legacy order still reproduces existing walls exactly: 12 walls of the
clean-wall study, including fixture and facade walls, came out identical.
Adopted in the unified recipe from v1.17 (2026-10-25). Every indoor wall now
gets fixtures and a closed room, whatever its crack style; v1.16 kept fixtures
to network walls. `--recipe unified_v1.16` rebuilds v1.16.0 walls exactly.
QA: 10 walls, all indoor walls with 2–4 fixtures in a closed room, every token
placed, contract clean (57 frames), ~17 s per wall.

### Clean-wall policy study (2026-10-23)

Which walls should stay crack-free? The same 200 unified-recipe walls (seeds
820000–820199) were rendered twice, cracked and crack-free. Each arm took
one render per wall, so the arms differ only in which walls are plain:

- **A, material-aware:** the shared table. 77 plain walls.
- **B, all cracked:** no plain walls.
- **C, uniform:** the same 77 plain walls in number, chosen by the same
  per-wall draw regardless of material.

Each arm trained from scratch on its own walls: 5 seeds × 4000 steps.
Scoring used protocol v2, with paired 95% intervals against A (photos or
source images and seeds resampled). Files:
`outputs/domain_transfer/syncrack3d_cleanpolicy_eval_v1/`. Study script:
`studies/syncrack3d_cleanpolicy_study.py`.

| real benchmark | A material-aware | B − A all cracked | C − A uniform |
|---|---:|---|---|
| BD3 crack IoU | 0.561 | +0.041 [+0.022, +0.058] | −0.028 [−0.047, −0.011] |
| BD3 centreline F1 | 0.874 | +0.027 [+0.013, +0.040] | −0.019 [−0.036, −0.003] |
| BD3 recall / precision | 0.803 / 0.960 | +0.071 / −0.029 | −0.030 / −0.001 (n.s.) |
| BD3 plain-wall alarms | 1.6% | +2.4 pp [+0.7, +4.3] | +0.6 pp (n.s.) |
| CUBIT crack-photo F1 | 0.872 | −0.011 [−0.022, −0.001] | +0.004 (n.s.) |
| CUBIT precision | 0.869 | −0.048 [−0.065, −0.036] | −0.002 (n.s.) |
| CUBIT spalling photos with false crack | 68% | 90% | 71% |
| BuildCrack vote F1 | 0.687 | +0.006 (n.s.) | −0.044 [−0.063, −0.028] |
| BuildCrack vote precision | 0.856 | −0.033 [−0.053, −0.014] | +0.016 [+0.001, +0.031] |
| BuildCrack empty-tile alarms | 27% | 33% | 33% |

**Material-aware (A) is the best balance.** The two alternatives fail in
different ways:

- **All cracked (B)** finds more cracks on BD3: recall +0.071, F1 +0.027.
  But it raises false alarms on every benchmark, so it is a recall–precision
  trade, not a gain:
  - BD3 plain walls 1.6% → 3.9%;
  - CUBIT spalling photos with a false crack 68% → 90%;
  - CUBIT and BuildCrack precision −0.03 to −0.05.

  B also had more cracked views (997 vs 627) and fewer views overall (1,210
  vs 1,529), because crack-free walls export more views. Its gain is partly
  more positives.
- **Uniform (C)** loses to A on BD3 (F1 −0.019) and BuildCrack (vote F1
  −0.044), and ties on CUBIT, although only 20 of 200 walls differ from A.
  Uniform plain walls fall on paint and stucco as often as on patterned
  masonry. That removes crack views on paint, where most real cracks are,
  without adding negatives where false alarms happen (grout, joints).

The synthetic eval set agrees. Fixture-negative alarms were A 7.7%, B 13.4%
and C 10.5%; on stone block, A 15% and B 54%.

Caveats: one training size (200 walls, each arm from scratch) and five
seeds. BuildCrack per-seed spread is large (C 0.15), so the vote intervals
are the firmer evidence. v1.9.0, trained on more data (2,243 views), still
scores higher on BuildCrack (vote F1 0.723).

### Matched indoor finishes study (2026-10-24)

`--shell-finish matched` changes the indoor context. Side walls carry the test
wall's own finish in 70% of rooms (a crack-free copy, bump only), otherwise a
different paint; the floor gets an Infinigen floor material. Outdoor facades
already continue the test wall's base material and are unchanged.

The study rendered the same 200 all-cracked unified walls as the clean-wall
study's control (`cleannone`). Cracks, cameras, lights and props were
verified identical wall by wall, so only the context differs:

- 100 indoor walls had side walls: 69 matched, 31 painted;
- 78 indoor walls had only a floor;
- 22 were outdoor and unchanged.

Training used 5 seeds × 4000 steps from scratch. Files:
`outputs/domain_transfer/syncrack3d_shellmatch_eval_v1/`.

| | plain shell | matched − plain [95%] |
|---|---:|---|
| BD3 centreline F1 | 0.901 | −0.012 [−0.027, −0.002] |
| BD3 crack IoU | 0.602 | −0.014 (n.s.) |
| BD3 plain-wall alarms | 3.9% | −0.1 pp (n.s.) |
| CUBIT crack-photo F1 | 0.861 | −0.004 (n.s.) |
| CUBIT spalling photos with false crack | 90% | 89% |
| BuildCrack vote F1 | 0.694 | +0.033 [+0.023, +0.044] |
| BuildCrack vote recall | | +0.049 [+0.034, +0.064] |
| BuildCrack empty-tile alarms | 33% | 27% |
| synthetic eval set F1, all cracked subsets | 0.933 | +0.003 |
| render time, 30 paired walls | 567 s | 566 s (×0.999) |

**What changed:**

- **BuildCrack improves.** Its tiles show real building context. With 200
  walls the matched arm reaches v1.9.0's vote F1 (0.727 vs 0.723), which was
  trained on 2,243 views.
- **BD3 F1 dips slightly.** The interval only just excludes zero, and IoU and
  plain-wall alarms are unchanged.
- **CUBIT ties.**
- **Render time is unchanged.** Side walls use bump relief only, and
  everything averages 19 s per wall (2.8 s per view at 640 px, 64 samples).

The effect is diluted: 78 of the 178 indoor walls had no side walls, so only
their floor changed.

Known rare case, also present in v1.8.0: a facade close-up can sit just
behind the context downpipe and render almost black (2 of 445 v1.8.0 facade
views).

## Reproduce the release data

`--recipe` sets every flag to the one a release batch was rendered with;
flags given explicitly override it (`syncrack3d/recipes.py`). Pick the
material with `--material-category`, or leave it to `--wall-materials
diverse`.

| recipe | walls |
|---|---|
| `legacy` | indoor walls, legacy cracks; some walls plain by material (the v1.0.0–v1.3.0 base walls) |
| `major` | indoor walls, broad open and branched cracks |
| `network` | indoor paint and stucco walls with crack networks and indoor structures (v1.7.0 on) |

The v1.8.0 `facade` recipe (outdoor facades) is gone with the outdoor scene in
v1.22.0; check out the `v1.21.0` tag to rebuild those walls.

```bash
python syncrack3d/build_dataset.py --recipe network --seeds 176022-176023 \
  --material-category stucco_neutral --output out/network --resume
```

A seed with the same recipe and material rebuilds the same wall, cameras and
cracks as the release. The training set also kept a selected subset of views
per batch: `releases/*.json` lists them with their seeds.

Renders are deterministic: the same command gives identical images.

From v1.9.0, props are placed by their real bounds, not by a catalogue
footprint (`--prop-placement measured`, the default). The old placement let
furniture clip into the scene: in 40 indoor test walls with `--max-props 3`,
35 of 58 props overlapped something. Rugs went up to 2.5 m through the test
wall and shelves 23 cm. Props also crossed the side returns, overlapped each
other and fixtures, and food boxes sank up to 13 cm into the floor. Five
props, mostly shelves, held a camera inside them, so those views showed the
inside of a shelf.

Props now stand loose on the floor, the only placement:

- turned any way, back face from 2 cm past the wall relief out to 0.9 m;
- on the floor, between the side returns and under the ceiling;
- stepped out from the wall past other props, fixtures, skirting, lights and
  cameras;
- dropped if they would hide more than half of a crack token face-on, or if
  no spot fits.

A camera left inside a prop is dropped. Rebuilding a wall from before v1.18
moves any prop that is in view.

Fixtures likewise follow one rule (v1.20.0): each kind's size, anchor height
and hardware come from a table in `structures.py`, and every fixture's built
extent stays on the wall and 3 cm clear of every other; a door and the trims
go first, and one that finds no room is dropped.

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

# 4. score a crack model (protocol v2, see below). Checkpoints are runs/s*/last.pt under --model.
#    BD3: 487 minor_crack + 355 plain (eval/bd3/README.md). predict.py writes <model>/pred_v2/s*/.
python eval/bd3/predict.py --model path/to/checkpoint_dir
python eval/bd3/score.py --pred path/to/checkpoint_dir/pred_v2/s0
#    CUBIT-InSeg at 1/3 scale (1500x1000). Crack photos and spalling false alarms.
#    Data: --data, or $SYNCRACK3D_CUBIT / $SYNCRACK3D_BUILDCRACK. Neither dataset is included.
python eval/cubit/score.py --model path/to/checkpoint_dir --data path/to/CUBIT-InSeg --out path/to/cubit_out
python eval/cubit/compare.py --a path/to/cubit_a --b path/to/cubit_b --out compare.json
#    BuildCrack tiles, grouped by source photo. Pass one --model per checkpoint.
python eval/buildcrack/score.py --model v1.8.0=path/to/checkpoint_dir --data path/to/BuildCrack --out path/to/buildcrack_out
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
| `--crack-path-mode straight`, `--crack-axis-snap` | near-straight cracks. `--crack-axis-snap` has no effect: it turns the card, and the field never read the card's spin (see `--crack-orientation`) |
| `--crack-orientation random` | each crack runs at its card's spin angle. By default (`vertical`, every release to v1.7.0) a crack runs along the wall's vertical with lateral wander: 79% of v1.2.0–v1.3.0 crack segments are near-vertical and 3–4% near-horizontal, against 31–49% near-horizontal in BD3, BuildCrack and CUBIT. Training rotates images by multiples of 90°, which hides most of this |
| `--crack-junction-fraction` | share of cracks moved to structure edges (needs `--structures`): diagonal from the corners of windows, doors, vents and access panels, under window sills and beside door jambs. Junction cracks set their own direction; edge cracks get a near-straight path; a moved crack keeps the plan's spacing (`syncrack3d/junctions.py`). Unified recipe: .5 from v1.24, 0 before |
| `--light-profile hard_mix`, `--relight-per-view` | sun / bare bulb / raking torch among the regimes; a new light rig per view |
| `--camera-isp`, `--awb-strength`, `--exposure-ev-range` | phone-camera processing, partial white balance, exposure spread |

Seeds are reproducible per wall: a wall's material, cracks, cameras and lights
depend only on its seed, not on which seeds ran before it in the same process.

## Evaluation protocol

Protocol v2 is the standard from v1.7.0 on. All three scorers use it by default.

- **Test-time upscaling 1.5x.** Each photo is upscaled 1.5x (bicubic) before
  inference and the class probabilities are resized back, so labels and
  predictions stay at the benchmark's resolution (`eval/infer.py`,
  `TEST_SCALE`). Real hairlines are 1–2 px, thinner than the training renders.
  The factor was chosen on CUBIT (F1 rose to 1.5x and was flat at 2x) and then
  checked on BuildCrack (+0.044 F1 on v1.3.0) and BD3 (IoU 0.573 -> 0.579).
- **Centreline tolerance 5 px** (`eval/metrics.py`, `TOLERANCE_PX`). The
  tolerance was raised from 3 px after the v1.0.0–v1.5.0 results. Human crack
  polygons are wide and their centrelines uncertain (median labelled width
  about 8.6 px on BuildCrack). From 1 to 5 px the absolute F1 changes (by up to
  0.06 on BuildCrack), but the difference between versions moved by at most
  0.004 and no ranking changed (`outputs/domain_transfer/syncrack3d_tolerance_check_v1.json`).
- Result files record `protocol: {tolerance_px, test_scale}`, and
  `eval/cubit/compare.py` refuses to compare runs scored under different
  protocols.

The per-version sections above (v1.0.0–v1.7.0) report protocol v1: 3 px,
no upscaling. To reproduce them, pass `--tolerance 3 --scale 1` to
`cubit/score.py` and `buildcrack/score.py`, `--tolerance 3` to `bd3/score.py`,
and `--scale 1` to `bd3/predict.py`. The metric key is now `centreline_f1`
(it was `centreline_f1_3px`).

Every release re-scored under protocol v2 (mean of five seeds; BuildCrack also
as the five-seed vote). Files: `outputs/domain_transfer/syncrack3d_eval_v2/`.

| version | BD3 F1 | BD3 IoU | BD3 plain alarms | CUBIT CR1 F1 | CUBIT spalling false crack | BuildCrack F1 | BuildCrack vote F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1.2.0 | 0.890 | 0.566 | 0.008 | 0.859 | 64% | 0.585 | 0.583 |
| v1.3.0 | 0.896 | 0.579 | 0.008 | 0.871 | 63% | 0.595 | 0.599 |
| v1.4.0 | 0.901 | 0.591 | 0.016 | 0.870 | 75% | 0.657 | 0.667 |
| v1.5.0 | 0.901 | 0.588 | 0.010 | 0.871 | 72% | 0.645 | 0.651 |
| v1.6.0 | 0.904 | 0.593 | 0.015 | 0.871 | 81% | 0.689 | 0.704 |
| v1.7.0 | 0.909 | 0.597 | 0.016 | 0.875 | 80% | 0.696 | 0.710 |

Upscaling raises false cracks on CUBIT spalling photos (v1.7.0: 72% under
protocol v1, 80% under v2); BD3 plain-wall alarms stay below 2%.

## Major cracks (experimental)

`--crack-profile standard` draws the ordinary crack types.
`--crack-profile wide` draws wide open and wide branched cracks equally;
`--crack-profile mixed` chooses wide or standard with probability 0.5 per crack.
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
an orthographic Position-pass render of the wall alone, colour, roughness,
normal map, class coverage) and the patches become displaced USD meshes with
semantic labels. Every sample whose ray hits the wall counts, however deep:
masonry joints reach several centimetres. The Isaac scripts need Isaac Sim 5.0.

```bash
python isaac_port/port_wall.py --seed 112000 --material-category paint_neutral --out out/port --blender-only
python isaac_port/port_wall.py --out out/port --isaac-only --isaac-py path/to/isaac-sim/python   # or $SYNCRACK3D_ISAAC_PY
```

### Whole rooms for embodied inspection

Close-ups are posed at the cracks, so a policy trained on them never has to
search. For embodied work, export the whole room instead, with no cameras:

```bash
python syncrack3d/build_dataset.py --recipe unified --seeds 910004 --output out/rooms --no-render --save-blend
python isaac_port/export_room.py out/rooms/seed_910004 out/isaac/seed_910004            # Blender: bakes
OMNI_KIT_ACCEPT_EULA=YES path/to/isaac-sim-5/python isaac_port/compose_room.py out/isaac/seed_910004 --preview
```

Open `scene.usda` in Isaac Sim 5. It is in metres with Z up and the same
frame as the Blender scene: test wall front at y = 0 facing -Y, floor at
z = 0. Rooms are indoor only, as the generator is from v1.22.

- **Test wall:** baked over its full face at about 0.6 mm per texel (capped at
  8192 px). Its height is the material's own displacement vector projected on
  the wall normal, so tile relief and cracks are both kept. The mesh holds the
  relief to 2.5 mm. Finer relief and the host bump are in the normal map.
- **Labels:** wall faces are split into `crack` and `wall` prims. Furniture,
  fixtures (skirting included), floor and ceiling carry their own semantic
  class labels.
- **Other surfaces:** only the test wall keeps real relief. Displacement
  elsewhere, such as the window sill and head that share the wall's material
  (v1.23), is baked into the normal map.
- **Ground truth:** `gt.npz` holds 1 mm crack coverage for reward and
  evaluation. It is never a policy input. The `crack` prim covers any 2.5 mm
  cell a crack touches, so its rendered masks are slightly wider than the
  crack itself.
- **Room:** procedural props are baked to textures. A back wall closes the
  shell, and the room is lit by its own lamps (`--open-room` keeps the open
  side and adds a dome light). `scene_meta.json` gives the room bounds, the
  defects in world coordinates and a spawn pose clear of furniture.
- **Rendering:** previews are path traced. Real-time RTX needs a few frames to
  settle after a new camera; its first frame shows blotchy dark patches on the
  wall.

Only the test wall is cracked. The other surfaces are plain.

#### Detector false alarms at room scale

`render_room_views.py` renders embodied-style views of a composed room: 60
poses per room, both path traced and real-time. 70% of poses look at the test
wall from 0.4–2.5 m at up to 35° off its normal; the rest look anywhere.
`--hard-light` dims the room's area lights and adds two small bright lamps.
`--shadow-only` writes cast-shadow masks for the same poses (lamp-only renders
with and without furniture and fixtures).

`room_false_alarms.py` passes each render through the training camera model
and runs a detector (protocol v2: 1.5× upscale, 5 px tolerance, majority of
seeds). It assigns each false-alarm blob a cause: cast shadow, object edge,
shading edge, paint/material, or relief/other.

Results for the v1.8.0 checkpoint on 3 rooms × 60 poses (2026-10-22), path
traced:

| lighting | views with a false alarm | crack-free views with one | blobs per view | visible cracks found |
|---|---|---|---|---|
| room lamps (soft) | 42% | 36% | 1.42 | 31% (14% under 150 px, 4/4 over 600 px) |
| hard lamps | 37% | 29% | 1.04 | 30% |

- **Object edges cause most alarms.** Under soft light they are 84% of
  false-alarm pixels: thin dark lines such as wall cables, panel-seam
  sealant, pipe edges, and the line where a box meets the floor. Tile grout
  and marble veins make up most of the rest.
- **Long cast-shadow edges do not trigger the detector.** Under hard light it
  fires on 0.1% of shadow-boundary pixels more than 5 px from any object
  outline (290k pixels). It fires on 1.3% of those near outlines: the shadow
  line beside a pipe, contact shadows under props, wall–ceiling junctions.
  These are 10 of 188 false-alarm blobs.
- **Real-time renders** find more cracks (51% under soft light) at a similar
  false-alarm rate.

## Maintainers

This folder is a snapshot of the repository generator, then edited here.
`controlled_wall_generator/` and `infinigen/` keep paint peeling and blistering,
including the two-layer peel. Rebuilding this app from the repo with
`python build_from_repo.py` would put that code back and overwrite the
crack-only edits; do that only after those edits have been ported, and pass
`--overwrite-local-edits` if you mean to discard them.

## Crack networks (v1.4.0)

Use `--crack-profile crazing` for map cracking: connected, irregular
closed cells and shared junctions across each defect patch. A smoothly warped
2-D Voronoi edge field drives both the shallow groove and its crack AOV.
This is a procedural appearance model, not a fracture simulation.

From this repository's root, with the usual Blender Python environment:

```bash
python -m syncrack3d.build_dataset --seeds 190010-190019 --output out/networks \
  --wall-materials diverse --material-category paint_neutral --clean-wall-scale 0 \
  --crack-profile crazing --crack-network-edge organic --crack-network-extent 1.4 \
  --crack-network-spacing .02 .06 \
  --crack-width-scale 2 --crack-label-mode support \
  --defect-size-range .3 .45 --min-defects 1 --max-defects 2 --max-props 0 \
  --resolution 1024 --samples 64 --dicing-rate 1 --wall-normal-mode explicit \
  --closeup-fill .9 1.1 --closeups-per-defect 2
```

For a mixture, keep `--crack-profile standard` (or `wide`/`mixed`) and add
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
record the sampled spacing, warp, seed offset, width, depth, edge and extent.

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

`--crack-network-edge organic` is the default. The outline is a rounded
square pushed inwards by low-frequency noise, and the crack lines narrow to
nothing in a band near it, so the web ends in tapering cracks instead of
along four straight edges. The outline comes from its own random stream, so
a seed's cells, widths, depth and staining are unchanged.
`--crack-network-extent` scales only that field (default 1.4). Extent 1
keeps about 57% of a square patch's crack length; 1.4 keeps about 95%, and
that is what ties the square networks in the v1.7.0 study. Pass
`--crack-network-edge square --crack-network-extent 1` to reproduce the
v1.4.0–v1.5.0 patches.

Validation: 33 tests pass. The final controlled-wall pilot (seeds
190000–190002, 512 px) passes the label contract on six views, with two
close-ups retained. A 1024 px textured painted-wall check (seed 190010,
width scale 2, dicing rate 1) shows visible faint networks and passes the
contract on both views; its close-up is retained. A six-instance mixed
smoke scene (seed 190020, fraction .5) contains three crazing networks and
three ordinary cracks. These are generator checks, not transfer results.

## Structural background details (v1.5.0)

`--structures auto` adds a selection of background assemblies. As of
v1.21.0: a door, framed windows with sills and lintels, vents, access panels,
utility cabinets, and full-length runs (cables in clips, conduit, pipes,
plumbing, trunking, sealant panel joints). The v1.5.0 list below it (ledges,
gutters, outdoor kinds) is history; v1.22.0 removed the outdoor scene.
v1.23.0 gives those fixtures ranged finishes and a fine bump, and builds the
window sill and head from the inspection wall.
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
  --structures auto --structure-views 2 \
  --structure-clean-fraction .35 --wall-materials diverse \
  --min-defects 1 --max-defects 3 --max-props 0 \
  --resolution 640 --samples 64 --wall-normal-mode explicit \
  --crack-label-mode support

python tools/prepare_defect_annotated_dataset.py -i out/structures/frames \
  -o out/structure_pack --aov-masks on --instance-masks on --with-bboxes \
  --max-negative-fraction .5
python tools/check_instance_contract.py --pack out/structure_pack
```

`--structures services` restricts sampling to cables, conduit, pipes and boxes.
`all` attempts every
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
