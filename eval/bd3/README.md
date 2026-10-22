# BD3 crack-only eval

Frozen subset of BD3 v2 (`outputs/domain_transfer/bd3_full_real_cv_v2`). Peeling photos are not included.

| folder | photos | role |
|---|---:|---|
| minor_crack | 487 | crack labels (class 1). `minor_crack_0021` is absent; its annotation was empty |
| plain | 355 | clean walls, all-background masks, scored as negatives |

842 photos, 512×512. Labels are 0 background, 1 crack, 255 ignore. These photos were inspected while the generator was designed, so this is a development set, not an untouched holdout. Near-duplicate groups are in `manifest.json` (`group`). Do not split a group across train and test.

Score a directory of `<id>.png` class maps (crack = 1; class 2 is not counted as crack):

```bash
python eval/bd3/predict.py --model path/to/checkpoint_dir   # writes <model>/pred_v2/s*/ with 1.5x upscaling
python eval/bd3/score.py --pred path/to/checkpoint_dir/pred_v2/s0
```

Minor-crack metrics are pooled centreline F1 at 5 px (protocol v2; `--tolerance 3` gives the v1 numbers), clDice and crack IoU (`eval/metrics.py`). A plain photo is a false alarm when predicted crack pixels exceed 0.1% of the image.
