#!/bin/bash
# Screen: SynCrack3D v1.20.0 with combined plaster and stucco, without the outdoor dirt (wall weathering: rain
# streaks, ground splash, algae; and rain-wash staining under facade cracks), against v120mc at 250 walls (seeds
# 910000-910249), U-Net R34, 4,000 steps, five seeds, zero-shot BD3 / CUBIT / BuildCrack. Only the 22 outdoor walls
# change; the 228 indoor walls ($D/unchanged.json) are hard links to the v120mc renders.
# The code is v120mc's ($D/code_diff.patch) minus the weathering code.
#   setsid nohup eval/studies/syncrack3d_no_dirt_250.sh >> /mnt/nvme_storage/syncrack3d_v1.20.0-nodirt_dataset/run250.log 2>&1 < /dev/null & disown
set -u
APP=${APP:-$(cd "$(dirname "$0")/../.." && pwd)}  # this repository
CODE=/mnt/nvme_storage/syncrack3d_v1.20-nodirt_code
OLD=/mnt/nvme_storage/syncrack3d_v1.20.0-matcombine_dataset/train_a/walls
D=/mnt/nvme_storage/syncrack3d_v1.20.0-nodirt_dataset
C=$D/train_a
BPY=/home/shiven/anaconda3/envs/infinigen_v2/bin/python
PY=/home/shiven/anaconda3/bin/python
S="nice -n 5 env PYTHONPATH=$APP/eval $PY -m studies.syncrack3d_v116_study"
mkdir -p $C/walls/frames
git -C $CODE rev-parse HEAD > $D/code_commit.txt
git -C $CODE diff > $D/code_diff.patch

if [ ! -f $C/done ]; then
  for s in $($PY -c "import json; print(' '.join(map(str, json.load(open('$D/unchanged.json')))))"); do
    [ -e $C/walls/seed_$s ] || cp -al $OLD/seed_$s $C/walls/seed_$s
    [ -e $C/walls/frames/seed_$s ] || cp -al $OLD/frames/seed_$s $C/walls/frames/seed_$s
  done
  echo "=== linked $(ls -d $C/walls/seed_* | wc -l) walls, render $(date -Is)"
  redo=$($PY -c "
import os
print(' '.join(str(s) for s in range(910000, 910250) if not os.path.exists('$C/walls/seed_%d/scene.json' % s)))")
  set -- $redo
  for half in 0 1; do
    seeds=$(i=0; for s in "$@"; do [ $((i % 2)) = $half ] && echo $s; i=$((i + 1)); done | paste -sd,)
    [ -n "$seeds" ] && (cd $CODE && for try in 1 2 3; do  # a crashed Blender resumes where it stopped
       $BPY syncrack3d/build_dataset.py --recipe unified --seeds $seeds --output $C/walls \
         --keep-going --resume >> $C/render_$half.log 2>&1; done) &
  done
  wait
  missing=$($PY -c "
import os
print(sum(not os.path.exists('$C/walls/seed_%d/scene.json' % s) for s in range(910000, 910250)))")
  [ "$missing" = 0 ] || { echo "$missing walls missing, stopping"; exit 1; }
  (cd $CODE && rm -rf $C/pack
   $BPY tools/prepare_defect_annotated_dataset.py -i $C/walls/frames -o $C/pack --aov-masks auto --instance-masks auto \
     --with-bboxes --max-negative-fraction 1 >> $C/export.log 2>&1
   $BPY tools/check_instance_contract.py --pack $C/pack > $C/contract.txt 2>&1; echo "contract_train_a=$?")
  echo "$(ls $C/pack/images | wc -l) views, $(ls -d $C/walls/seed_* | wc -l) walls; 910000-910249" > $C/done
  echo "=== done train_a $(cat $C/done) $(date -Is)"
fi

cd $APP
STUDY_TAG=v120nd $S train unet_w250 --walls 250 --arch unet_r34 --seeds 0 1 2 3 4 --iters 4000
STUDY_TAG=v120nd $S eval unet_w250 --splits --real
echo NO_DIRT_250_DONE $(date -Is)
