#!/bin/bash
# SynCrack3D v1.17.0 with loose props (--prop-placement loose) against the tested v1.17.0, same test as
# syncrack3d_v2_compare.sh (1,000 walls, U-Net R34, 4,000 steps, seeds 0-2, zero-shot BD3 / CUBIT / BuildCrack).
# Props are placed last and nothing random is drawn after them, so a wall with no prop under either rule
# re-renders pixel-identically: those are hard links to the v1.17.0 renders, the rest are rendered.
# Which walls get a prop comes from a scene-only build (--no-render). Resumable.
#   setsid nohup studies/syncrack3d_loose_props.sh >> /mnt/nvme_storage/syncrack3d_v1.17.0-looseprops_dataset/run.log 2>&1 < /dev/null & disown
set -u
APP=${APP:-$(cd "$(dirname "$0")/.." && pwd)}  # this repository
CODE=/mnt/nvme_storage/syncrack3d_v1.17-looseprops_code
SRC=/mnt/nvme_storage/syncrack3d_v1.17.0_dataset
D=/mnt/nvme_storage/syncrack3d_v1.17.0-looseprops_dataset
BPY=/home/shiven/anaconda3/envs/infinigen_v2/bin/python
PY=/home/shiven/anaconda3/bin/python
S="nice -n 5 env PYTHONPATH=$APP $PY -m studies.syncrack3d_v116_study"
FLAGS="--recipe unified --prop-placement loose"
mkdir -p $D
git -C $CODE rev-parse HEAD > $D/code_commit.txt
git -C $CODE diff > $D/code_diff.patch

for spec in train_a:910000-910499 train_b:910500-910999; do
  name=${spec%%:*}; seeds=${spec#*:}; C=$D/$name
  [ -f $C/done ] && continue
  echo "=== $name plan $(date -Is)"
  (cd $CODE && $BPY syncrack3d/build_dataset.py $FLAGS --seeds $seeds --output $C/plan --no-render --keep-going --resume \
     >> $C.plan.log 2>&1)
  redo=$($PY -c "
import json, os, sys
lo, hi = map(int, '$seeds'.split('-'))
redo = []
for s in range(lo, hi + 1):
    new, old = '$C/plan/seed_%d/scene.json' % s, '$SRC/$name/walls/seed_%d/scene.json' % s
    if not (os.path.exists(new) and os.path.exists(old)) or json.load(open(new))['props'] or json.load(open(old))['props']:
        redo.append(str(s))
print(','.join(redo))")
  mkdir -p $C/walls/frames
  for w in $SRC/$name/walls/seed_*; do
    s=$(basename $w)
    case ",$redo," in *,${s#seed_},*) continue;; esac
    [ -e $C/walls/$s ] || cp -al $w $C/walls/$s
    [ -e $C/walls/frames/$s ] || cp -al $SRC/$name/walls/frames/$s $C/walls/frames/$s
  done
  echo "linked $(ls -d $C/walls/seed_* | wc -l) walls; rendering $(echo $redo | tr ',' '\n' | wc -l): $redo"
  (cd $CODE && $BPY syncrack3d/build_dataset.py $FLAGS --seeds $redo --output $C/walls --keep-going --resume >> $C/render.log 2>&1
   rm -rf $C/pack
   $BPY tools/prepare_defect_annotated_dataset.py -i $C/walls/frames -o $C/pack --aov-masks auto --instance-masks auto \
     --with-bboxes --max-negative-fraction 1 >> $C/export.log 2>&1
   $BPY tools/check_instance_contract.py --pack $C/pack > $C/contract.txt 2>&1; echo "contract_$name=$?")
  echo "$(ls $C/pack/images | wc -l) views, $(ls -d $C/walls/seed_* | wc -l) walls" > $C/done
  echo "=== done $name $(cat $C/done) $(date -Is)"
done

cd $APP
STUDY_TAG=v117lp $S train unet_w1000_it4000 --walls 1000 --arch unet_r34 --seeds 0 1 2 --iters 4000
STUDY_TAG=v117lp $S eval unet_w1000_it4000 --splits --real
echo LOOSE_PROPS_DONE $(date -Is)
