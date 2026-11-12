# SynCrack3D

SynCrack3D is a synthetic scene synthesiser that can generate synthetic images of cracked walls with pixel-exact labels.

Each wall is built in Blender, its cracks are real displaced geometry in the wall material, and every view is rendered with Cycles. For each view you get the RGB image, depth image, a crack mask, instance masks, wall relief map, boxes and COCO annotations.

One seed gives one wall. The same seed always gives the same wall, cameras and lights, whichever other seeds run in the same process.

## Layout

```
syncrack3d/        generator: wall, materials, cracks, fixtures, cameras, lighting, render
cwg_infinigen/     modified subset of Infinigen (materials, crack injection, labels, sensor model, props)
tools/             export to a training pack, and a label check
isaac_port/        export to Isaac Sim (USD)
eval-syncrack3d/   submodule: evaluation scorers
```

`eval-syncrack3d` is a separate repository, private for now. With access, clone
with `git clone --recurse-submodules`, or run `git submodule update --init
eval-syncrack3d` in an existing checkout. The generator, export and Isaac port
work without it; only the detector diagnostics `isaac_port/room_false_alarms.py`
and `tools/render_light_compare.py` use its scorer code.

## Setup

Add the Python environment:

```bash
# generator: Python 3.11 (bpy 4.2 needs it); Blender runs as a Python module
pip install -r requirements.txt
python -m pytest syncrack3d/tests          # 85 tests, a few seconds
```

Run every command from the repository root. Cycles renders on the first GPU it
finds (OptiX, CUDA, HIP, Metal, oneAPI) and otherwise on the CPU. Blender prints
`Error: Not freed memory blocks` when a run exits; it is harmless.

## 1. Generate walls

```bash
python syncrack3d/build_dataset.py --recipe unified --seeds 0-99 --output out/walls --resume
```

`--recipe unified` is the recommended setting. Every wall is an indoor room wall
with fixtures (doors, windows, vents, pipes, cables, panel seams), at most one
prop, and one to three cracks. Each wall draws one of 10 materials (plaster, stucco,
concrete, brick, tile, large tile, marble, stone block, cobble wall,
sandstone) and one crack style:

| style | share | cracks |
|---|---|---|
| standard | about 0.72 | hairline to moderate, branched or double-run |
| wide | about 0.17 | open and wide branched cracks |
| crazing | about 0.11 | fine crack networks, on plaster and stucco only |

About half of the cracks start at a corner or edge of a window, door, vent or
access panel. Each crack gets two close-up views, and each wall also gets wider
context views. Lighting varies per wall; exposure and white balance per view.

Output per wall:

```
out/walls/seed_<seed>/scene.json             wall, material, cracks, fixtures, cameras, lights, settings
out/walls/seed_<seed>/defect_instances.json  one record per crack: style, width, depth, status
out/walls/frames/seed_<seed>/Image/          RGB views
out/walls/frames/seed_<seed>/DefectCrack/    crack coverage per view
out/walls/frames/seed_<seed>/DefectInstance*/ instance passes, used by the export
out/walls/frames/seed_<seed>/Depth/          camera-axis depth, uint16 PNG, value * 0.1 mm; 0 = no surface
out/walls/frames/seed_<seed>/WallRelief/     offset from the wall plane, uint16 PNG, (value - 32768) * 0.01 mm
out/walls/frames/seed_<seed>/Camera/         intrinsics, pose and the depth encodings
```

Useful options (`python syncrack3d/build_dataset.py --help` lists all):

| option | what it does |
|---|---|
| `--seeds 0-99` or `--seeds 3,7,12` | which walls to build |
| `--resume`, `--keep-going` | skip walls already built; continue past a failed wall |
| `--material-category brick` | force one material |
| `--crack-style-mix crazing=1` | force one crack style (`off`: use the crack flags below as given) |
| `--crack-profile`, `--min-defects`, `--max-defects`, `--defect-size-range` | crack type, number and size (with `--crack-style-mix off`) |
| `--crack-width-scale`, `--crack-depth-scale` | crack width (also changes the labels) and depth |
| `--crack-junction-fraction 0` | keep cracks away from fixture corners |
| `--crack-edge-profile original` | the smaller crack edges from before v1.29, none on crazing (unified: `strong`) |
| `--resolution`, `--samples` | image size and Cycles samples (unified: 640 px, 64 samples) |
| `--no-render --save-blend` | build and save the scene without rendering (for the Isaac export) |
| `--crack-render pbr` or `albedo` | ablation: crack as a normal map only, or as colour only |
| `--hide-context all` (or `fixtures`, `props`) | ablation: leave fixtures and/or props out of the render; layout and lights unchanged |
| `--no-export-depth` | skip the Depth, WallRelief and Camera files (written by default) |
| `--depth-only` | rebuild walls and render only Depth, WallRelief and Camera (1 sample), for walls rendered without them |

Depth and wall relief come from the same render as the RGB, from its first-sample
passes, so they measure the displaced geometry the image shows. To add them to walls
rendered before they existed, without re-rendering RGB or labels:

```bash
python tools/add_depth_to_dataset.py --dataset <dataset> --chunks test val train_a train_b \
  --build-flags "--recipe unified"
```

It rebuilds each wall from its seed, keeps it only if the rebuilt `scene.json` equals the
released one, and adds the files to the wall's frames and to the pack.

## 2. Export a training pack

```bash
python tools/prepare_defect_annotated_dataset.py -i out/walls/frames -o out/pack \
  --aov-masks on --instance-masks on --with-bboxes --max-negative-fraction 1
python tools/check_instance_contract.py --pack out/pack
```

`out/pack` holds `images/`, `masks/` (0 background, 60 crack), `ignore_masks/`,
`instance_masks/`, `instances/`, `bboxes/`, `bboxes_yolo/`,
`annotations_coco.json`, `class_names.txt`, `splits/`, and `depth/`, `wall_relief/` and
`cameras/` (not for walls built with `--no-export-depth`). Splits are by wall, so
no wall is in two splits. The contract check confirms that masks, instances and
boxes agree.

## 3. Export to Isaac Sim

Cracks exist as shader displacement at render time, so the Isaac Sim 5 export
bakes the wall's colour, roughness, relief and crack coverage and rebuilds the
wall as a displaced, labelled USD mesh.

```bash
# close-up panels, one per close-up view
python isaac_port/port_wall.py --seed 112000 --material-category plaster --out out/port --blender-only
python isaac_port/port_wall.py --out out/port --isaac-only --isaac-py path/to/isaac-sim/python

# a whole room, with no cameras posed at the cracks
python syncrack3d/build_dataset.py --recipe unified --seeds 910004 --output out/rooms --no-render --save-blend
python isaac_port/export_room.py out/rooms/seed_910004 out/isaac/seed_910004    # add --wall-only for the wall bake alone
OMNI_KIT_ACCEPT_EULA=YES path/to/isaac-sim/python isaac_port/compose_room.py out/isaac/seed_910004 --preview
```

USD scenes are in metres with Z up. The test wall faces -Y at y = 0, and its
meshes are labelled `crack` and `wall`.

## Licence

SynCrack3D is Copyright (c) 2025-2026 Shiven Garg and is released under the BSD 3-Clause License (`LICENSE`).

`cwg_infinigen/` is a modified subset of Infinigen (Copyright (c) 2023, Princeton University), also BSD 3-Clause. That notice is kept unchanged in `LICENSE-INFINIGEN` and in the headers of the derived files. Other bundled code keeps the licence file next to it. See `NOTICE`.
