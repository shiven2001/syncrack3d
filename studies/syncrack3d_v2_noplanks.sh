#!/bin/bash
# SynCrack3D v2.0.0 without wood-plank inspection walls, same test as syncrack3d_v2_compare.sh (1,000 walls,
# U-Net R34, 4,000 steps, seeds 0-2, zero-shot BD3 / CUBIT / BuildCrack). The no-plank picker redraws only
# plank walls, and an unchanged wall re-renders pixel-identically, so the other walls are hard links to
# the v2.0.0 renders and only the former plank walls are rendered. Resumable.
#   setsid nohup studies/syncrack3d_v2_noplanks.sh >> /mnt/nvme_storage/syncrack3d_v2.0.0-noplanks_dataset/run.log 2>&1 < /dev/null & disown
set -u
APP=${APP:-$(cd "$(dirname "$0")/.." && pwd)}  # this repository
CODE=/mnt/nvme_storage/syncrack3d_v2.0.0-noplanks_code
SRC=/mnt/nvme_storage/syncrack3d_v2.0.0_dataset
D=/mnt/nvme_storage/syncrack3d_v2.0.0-noplanks_dataset
BPY=/home/shiven/anaconda3/envs/infinigen_v2/bin/python
PY=/home/shiven/anaconda3/bin/python
S="nice -n 5 env PYTHONPATH=$APP $PY -m studies.syncrack3d_v116_study"
mkdir -p $D
git -C $CODE rev-parse HEAD > $D/code_commit.txt
git -C $CODE diff > $D/code_diff.patch

for spec in train_a:910000-910499 train_b:910500-910999; do
  name=${spec%%:*}; C=$D/$name
  [ -f $C/done ] && continue
  echo "=== $name $(date -Is)"
  # Seeds whose v2.0.0 draw was wood planks, from the picker itself.
  redo=$(cd $CODE/syncrack3d && $BPY -c "
import importlib.util
spec = importlib.util.spec_from_file_location('md', '$SRC/../syncrack3d_v2.0.0_code/syncrack3d/material_diversity.py')
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
lo, hi = map(int, '${spec#*:}'.split('-'))
print(','.join(str(s) for s in range(lo, hi + 1) if md.sample_category(s) == 'wall_wood_planks'))")
  mkdir -p $C/walls/frames
  for w in $SRC/$name/walls/seed_*; do
    s=$(basename $w)
    case ",$redo," in *,${s#seed_},*) continue;; esac
    [ -e $C/walls/$s ] || cp -al $w $C/walls/$s
    [ -e $C/walls/frames/$s ] || cp -al $SRC/$name/walls/frames/$s $C/walls/frames/$s
  done
  echo "linked $(ls -d $C/walls/seed_* | wc -l) walls; rendering $redo"
  (cd $CODE && $BPY syncrack3d/build_dataset.py --seeds $redo --output $C/walls --keep-going --resume >> $C/render.log 2>&1
   rm -rf $C/pack
   $BPY tools/prepare_defect_annotated_dataset.py -i $C/walls/frames -o $C/pack --aov-masks auto --instance-masks auto \
     --with-bboxes --max-negative-fraction 1 >> $C/export.log 2>&1
   $BPY tools/check_instance_contract.py --pack $C/pack > $C/contract.txt 2>&1; echo "contract_$name=$?")
  echo "$(ls $C/pack/images | wc -l) views, $(ls -d $C/walls/seed_* | wc -l) walls, planks left: $(grep -l '"wall_wood_planks"' $C/walls/seed_*/scene.json | wc -l)" > $C/done
  echo "=== done $name $(cat $C/done) $(date -Is)"
done

cd $APP
STUDY_TAG=v200np $S train unet_w1000_it4000 --walls 1000 --arch unet_r34 --seeds 0 1 2 --iters 4000
STUDY_TAG=v200np $S eval unet_w1000_it4000 --splits --real
echo NOPLANKS_DONE $(date -Is)
