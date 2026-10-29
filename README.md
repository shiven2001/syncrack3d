# SynCrack3D

Procedural cracks on painted, plastered and masonry walls, rendered in Blender
Cycles with pixel-exact crack labels, instance masks, boxes and COCO
annotations. SynCrack3D generates cracks only. Version 1.18.0.

One seed is one wall: a wall material, cracks of one style, an indoor room or
an outdoor facade around it, wall fixtures, furniture, lighting, and several
close-up and context photographs with their labels. A wall depends only on its
seed, not on which seeds ran before it in the same process.

The development history (per-version results and studies) is in
[docs/HISTORY.md](docs/HISTORY.md).

## Setup

Two Python environments: one renders, one scores.

```bash
# generator: Python 3.11 (bpy 4.2 needs it); Blender runs as a Python module
pip install -r requirements.txt
python -m pytest syncrack3d/tests          # a few seconds

# scoring (any Python with CUDA PyTorch)
pip install -r eval/requirements.txt
```

Cycles renders on the first GPU it finds (OptiX, CUDA, HIP, Metal, oneAPI)
and otherwise on the CPU. Blender prints `Error: Not freed memory blocks`
when a run exits; it is harmless. Run commands from this folder.

## Generate, export, check

```bash
# 1. render walls (defaults are the SynCrack3D configuration)
python syncrack3d/build_dataset.py --seeds 0-99 --output out/walls --resume

# 2. export labels (0 background, 60 crack in masks/; scene-level train/val/test splits)
python tools/prepare_defect_annotated_dataset.py -i out/walls/frames -o out/pack \
  --aov-masks on --instance-masks on --with-bboxes --max-negative-fraction 1

# 3. check the instance/label contract
python tools/check_instance_contract.py --pack out/pack
```

At 640 px and 64 samples a wall takes about 21 s on an RTX 4090 and gives
about 6 views. `python syncrack3d/build_dataset.py --help` lists every option.
Each wall's `scene.json` records its material, crack style, fixtures, props,
lights, cameras and timing.

## How a wall is built

1. **Wall and material.** The inspection wall draws from Infinigen 2's wall
   picker (`wall_material_rand`, vendored under `syncrack3d/vendor`, BSD) with
   its own weights: mostly paint (about 39%, sometimes with a splat layer),
   then wall tile, stucco, brick, wood planks, stone-block masonry, concrete,
   painted brick, cobble and stone. Masonry and cobble are added to the
   picker, and so is painted stucco (about 14%: knockdown, sand float, skip
   trowel, popcorn and heavy orange peel), which Infinigen 2 lacks; it is
   built the way Infinigen 2's paint is (`syncrack3d/materials_v2/stucco.py`),
   with 0.5-6 mm of real displacement. Brick, block and tile courses stay
   horizontal. Each seed rebuilds the procedural graph, so the finish varies
   without an Infinigen install. Fixtures and props stand off the
   wall by its own measured relief (`syncrack3d/relief.py`). Shaders that draw their own cracks are left out so the
   only cracks in frame are labelled ones: `paint_flaked`, `wall_flaked_tile`,
   the cracked coat of `bricks_paint`, and the scratch layer. The older 15
   categories remain available through `--material-categories`, which draws
   uniformly from what it names. The wall then draws a crack style (below).
2. **Context.** Indoors, a room segment: floor, ceiling and two side walls.
   The side walls always carry a crack-free copy of the wall's own finish.
   The ceiling is Infinigen 2's ceiling picker (light paint,
   concrete, wood planks, wall tile); the floor is its floor picker (wood
   planks, carpet, concrete, indoor ground tile). Outdoors, a building facade
   whose ground is Infinigen 2's terrain picker (dirt, soil, sand, sandstone,
   granite, smooth stone, gravel concrete; `ground_cracked` and `stone` read as
   cracks, `mud` and `ice` as standing water, so they are left out), plus sky,
   sometimes
   an inside corner, an eave or a downpipe, and weathering. All of these
   surfaces are displaced as real geometry, not a bump map.
3. **Fixtures**, planned on their own random stream and built: switched
   sockets, doors, return vents, service hatches, trunking, plumbing,
   cables and conduits indoors; windows with sills, panel seams, downpipes,
   vents, hatches and lights outdoors. Indoor walls with a socket get up to
   two more. Finishes come from one palette per scene, all Infinigen 2:
   plastic services, pipes (copper, chrome, white or grey), hardware,
   cables, downpipes, window frames and doors. A window sill is always the
   wall's own material.

   Every fixture is placed by one rule with its own values
   (`structures.RULES`): a size, a height band for its bottom (a socket
   0.21-0.41 m, a window sill 0.85-1.05 m, a door on the floor), a gap kept
   below the ceiling, the built parts beyond its box (a window's sill, a
   door's casing), how far it stands off the wall, and a placement order.
   A draw is kept only if what it covers once built stays 5 cm from every
   fixture at an overlapping depth off the wall (a cable may cross a flush
   sealant seam, not a window); otherwise it is redrawn, and after 80 draws
   left out, so nothing clips. Cables, conduits, trunking and pipes are
   placed last, as routes from the floor to the ceiling or a side. Each plate is a UK double
   socket, two rockers above two three-pin outlets, so there is no
   separate switch. The plate is white, black, or satin metal. Doors are closed panel
   leaves set in a mitred architrave of the same wood (65-90 mm wide,
   24-30 mm proud of the wall, lapping over the leaf's edges). The leaf face
   sits 8 mm in front of the wall, behind the architrave, and its back half
   and back handle are removed because the wall is not cut. Service hatches, indoors and outdoors,
   are flush cabinet doors with one chrome cam lock. The cabinet is
   white, black, or satin metal.
   Windows are Infinigen casements mounted on the wall face, with opaque
   tinted glass so the uncut wall does not show through the panes, and a
   projecting sill with a metal drip under the frame.
   Utility boxes are powder-coated cabinets with a smoked window; their
   conduits run on to the edge of the wall. A wall cable or conduit is
   the same kind of route, one to three runs side by side, and it may
   turn before it gets there. Trunking is one of those routes as a duct,
   and plumbing is a pair of pipes on one route.    The service hatch is
   the same cabinet indoors and outdoors. The vent is the same
   return grille indoors and outdoors, with horizontal louvres, in
   white, black, or satin metal. An outdoor wall light is a
   matte cylinder open at both ends, and it casts warm light up and
   down the wall.
   Fixtures are weathered: painted and plastic parts are yellowed and
   grimy, and about half of the frames, boxes and trunking have a sealant
   bead where they meet the wall. Fixtures mount on the wall's displaced
   relief, not its flat plane, and props keep clear of it: the relief is
   measured on each wall's own material (`syncrack3d/relief.py`), up to
   about 8 cm for rough stone masonry, so nothing sinks into it.
4. **Crack tokens** around the fixtures. No crack centre falls under a large
   fixture (grown by 0.3 × the largest token size + 3 cm); 30% of tokens are
   tried 3–15 cm beside a cable, pipe or trunking run, so services partly
   cover some cracks. On facades, half the cracks are moved to fixture
   junctions: diagonal from the corners of openings, or near-straight along
   sills and trims.
5. **Crack injection** into the wall material: the crack field becomes real
   displaced relief and a coverage AOV for the labels (see below).
6. **Lighting and cameras.** Indoor light regimes: window raking, window soft,
   ceiling diffuse, lamp mixed, flash, with a weaker second source half the
   time. Outdoors: direct sun, overcast, shade or golden hour. Two close-ups
   per crack (phone lenses, up to 38° yaw and 26° pitch, off-centre aim) plus
   about one context view per four close-ups. A camera with another object
   within 0.35 m along its view axis is dropped.
7. **Props.** Each indoor wall is furnished as one room (kitchen 20%,
   bathroom 20%, living room 25%, bedroom 20%, utility 15%) with up to three
   items from that room's list (24 models). The
   bathroom, kitchen and fixture models are the vendored Infinigen versions. Every
   item is loose: it stands on the floor at any angle, its back 2 cm to
   0.9 m from the wall (beyond the wall's relief). Close-ups are taken
   0.16-0.45 m from the wall and context views 1-3 m, so an item further out
   would stand behind the cameras.

   Every model is scaled into real-world size ranges (`Prop.size_m`), and
   sofa, armchair and bed fabrics have their saturation damped to 55-75%.
   The shapes are Infinigen 1's; the surfaces are Infinigen 2's. Each part's
   material is replaced by one from the Infinigen 2 object picker for its
   role (upholstery, cabinet, table top, furniture wood, brushed steel for
   fridges and chrome, other metal, plastic, glass, ceramic), as bump on metre box-projected UVs. The
   furniture picker's cracked-paint layer is left out. Printed labels,
   plants and soil keep their Infinigen 1 material.

   Every item is placed by its measured bounds and must clear the wall,
   floor, ceiling, side walls, the far edge of the floor, fixtures,
   cameras and lights (0.15 m around each) and the other items.
   An item in the way of something steps further from the wall within the
   0.9 m band; otherwise a new position and angle are drawn, up to 40
   times, before it is dropped. Large furniture is dropped most often,
   because it would stand where a context camera is. An item may
   hide at most half of any crack seen face-on. A view then jammed against an
   item is dropped. Shower stalls are left out: cracks seen through their
   glass would be labelled background. Then the views are rendered with
   per-view exposure (+0.4 to +1.3 EV) and partial white balance.

### Crack styles

Each wall draws one style (`--crack-style-mix`). A style sets only what
defines that kind of crack; everything else is shared.

| style | share | walls |
|---|---|---|
| legacy | 0.65 | indoor; 2–3 cracks of 0.20–0.45 m: hairline, forked, moderate, branched, open (0.8–6 mm mouth) and double runs |
| major | 0.15 | indoor; 1–2 broad open and branched cracks, channel radius up to 12 mm, depth 8–18 mm |
| network | 0.10 | indoor paint or stucco; 2–3 crack networks (crazing): warped Voronoi cells 2–6 cm across with an irregular outline and stained channels |
| facade | 0.10 | outdoor; legacy cracks, half moved to fixture junctions, 60% with rain-wash dirt or efflorescence streaks below them |

### Crack field and labels

A crack is a polyline centreline with noise, a V-shaped channel profile with
tip taper and forks, and a height field that is added to the host material's
displacement, so the crack is real geometry under the light. On paint,
plaster and stucco 70% of cracks get a coating edge: torn paint lips and
1.5–4 mm chips, or a broken stucco finish coat. Brick, stone, concrete and
tile keep a single-material groove. The label is the channel's support (the
profile above 2% of its depth) plus the chips, written to a coverage AOV and
resolved per instance by the exporter.

## Options and ablation switches

Run controls: `--seeds`, `--output`, `--resolution` (640), `--samples` (64),
`--dicing-rate` (0.75), `--no-denoise`, `--no-render`, `--save-blend`,
`--keep-going`, `--resume`.

Every other option defaults to the configuration above and exists for data
design and ablations. Each switch changes only its own factor; the other
random streams are unchanged.

| factor | switch | ablation setting |
|---|---|---|
| crack morphology | `--crack-style-mix` | `legacy=1` (one style) |
| crack geometry | `--crack-render geometry\|pbr\|albedo` | bump map only, or colour only (labels identical) |
| coating edges | `--crack-layered-fraction` (0.7) | `0` |
| material diversity | `--material-categories CAT ...` | e.g. `paint_neutral paint_colour paint_colour_mid` |
| crack-free walls | `--clean-wall-scale` (0), `--clean-wall-uniform` | `1` (per-material table: paint and stucco 0.15, concrete 0.35, masonry, tile and stone 0.60), or one rate for every material |
| fixtures | `--no-fixtures`, `--crack-near-service-fraction` (0.3) | no fixtures (indoor rooms keep the sampled open shell) |
| indoor context | `--shell-finish matched\|plain` | `plain` (flat-colour side walls and floor) |
| props | `--max-props` (3) | `0` (bare room) |
| lighting | `--light-profile default\|hard_mix\|fixed`, `--relight-per-view` | `fixed`: one constant soft key light indoors; `hard_mix` adds sun, bare-bulb and raking-torch sources |
| camera processing | `--exposure-ev-range` (0.4 1.3), `--awb-strength` (0 0.5), `--camera-isp` | `0 0` and `1 1` (no exposure spread, full white balance); phone-like ISP |
| viewpoints | `--closeups-per-defect` (2), `--closeup-fill` (0.6 1.5), `--closeup-max-yaw` (38), `--closeup-max-pitch` (26), `--closeup-aim-jitter` (0.3), `--context-view-fraction` (0.25) | frontal, centred or close-up-only views |

## Evaluation

```bash
# Checkpoints are runs/s*/last.pt under --model.
# BD3: 487 minor_crack + 355 plain (eval/bd3/README.md). predict.py writes <model>/pred_v2/s*/.
python eval/bd3/predict.py --model path/to/checkpoint_dir
python eval/bd3/score.py --pred path/to/checkpoint_dir/pred_v2/s0
# CUBIT-InSeg at 1/3 scale (1500x1000): crack photos and spalling false alarms.
# Data: --data, or $SYNCRACK3D_CUBIT / $SYNCRACK3D_BUILDCRACK. Neither dataset is included.
python eval/cubit/score.py --model path/to/checkpoint_dir --data path/to/CUBIT-InSeg --out path/to/cubit_out
python eval/cubit/compare.py --a path/to/cubit_a --b path/to/cubit_b --out compare.json
# BuildCrack tiles, grouped by source photo. Pass one --model per checkpoint.
python eval/buildcrack/score.py --model unet=path/to/checkpoint_dir --data path/to/BuildCrack --out path/to/buildcrack_out
# Synthetic: build a frozen held-out set, then score on it.
python eval/synth/build.py --out out/evalset
python eval/synth/score.py --set out/evalset --model path/to/checkpoint_dir
```

- **Test-time upscaling 1.5×.** Each photo is upscaled 1.5× (bicubic) before
  inference and the class probabilities are resized back, so labels and
  predictions stay at the benchmark's resolution (`eval/infer.py`,
  `TEST_SCALE`). Real hairlines are 1–2 px, thinner than the training renders.
  The factor was chosen on CUBIT (F1 rose to 1.5× and was flat at 2×) and then
  checked on BuildCrack and BD3.
- **Centreline precision, recall and F1 at 5 px** (`eval/metrics.py`,
  `TOLERANCE_PX`). Both masks are thinned to 1 px skeletons; recall is the
  labelled crack length within 5 px of the prediction, precision the
  predicted length within 5 px of the label. Human crack polygons are wide
  and their centrelines uncertain (median labelled width about 8.6 px on
  BuildCrack). clDice and pixel crack IoU are reported beside it.
- **False alarms.** A crack-free photo or tile is an alarm when predicted
  crack pixels exceed 0.1% of it.
- Result files record `protocol: {tolerance_px, test_scale}`, and
  `eval/cubit/compare.py` refuses to compare runs scored under different
  settings.

## Can this simulate structural fracture?

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
python isaac_port/port_wall.py --out out/port --isaac-only --isaac-py path/to/isaac-sim/python   # or $SYNCRACK3D_ISAAC_PY
```

### Whole rooms for embodied inspection

Close-ups are posed at the cracks, so a policy trained on them never has to
search. For embodied work, export the whole room instead, with no cameras:

```bash
python syncrack3d/build_dataset.py --crack-style-mix legacy=1 --max-props 3 \
  --seeds 190531 --output out/rooms --no-render --save-blend
python isaac_port/export_room.py out/rooms/seed_190531 out/isaac/seed_190531            # Blender: bakes
OMNI_KIT_ACCEPT_EULA=YES path/to/isaac-sim-5/python isaac_port/compose_room.py out/isaac/seed_190531 --preview
```

Open `scene.usda` in Isaac Sim 5. It is in metres with Z up and the same frame
as the Blender scene: test wall front at y = 0 facing -Y, floor at z = 0.

- **Test wall:** baked over its full face at about 0.6 mm per texel (capped at
  8192 px). Its height is the material's own displacement vector projected on
  the wall normal, so tile relief and cracks are both kept. The mesh holds the
  relief to 2.5 mm. Finer relief and the host bump are in the normal map.
- **Labels:** wall faces are split into `crack` and `wall` prims. Furniture,
  fixtures, floor and ceiling carry their own semantic class labels.
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
and runs a detector (1.5× upscale, 5 px tolerance, majority of seeds). It assigns each false-alarm blob a cause: cast shadow, object edge,
shading edge, paint/material, or relief/other.

Results for an earlier SynCrack3D checkpoint (v1.8.0 in docs/HISTORY.md) on 3 rooms × 60 poses, path
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

## Licence notes

`cwg_infinigen/` is a vendored subset of [Infinigen](https://github.com/princeton-vl/infinigen)
(Princeton University, BSD 3-Clause; see `LICENSE-INFINIGEN`), with imports
renamed to `cwg_infinigen`. Infinigen 2.0.0a3 wall shaders and ProcFunc 0.37.0
are vendored under `syncrack3d/vendor/` (BSD-3-Clause; `LICENSE-INFINIGEN2`,
`LICENSE-PROCFUNC`). OcMesher,
mesh_to_sdf, and font licences stay beside those trees.
