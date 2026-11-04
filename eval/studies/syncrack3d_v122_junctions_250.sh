#!/bin/bash
# Screen: SynCrack3D v1.22.0 with junction cracks (--crack-junction-fraction .5: each crack token independently
# moves, with probability .5, to a corner or edge of a window, door, vent or access panel) against v122 at 250
# walls (seeds 910000-910249), U-Net R34, 4,000 steps, five seeds, zero-shot BD3 / CUBIT / BuildCrack. Same frozen
# code and seeds as v122; the junction step draws from its own stream, so only the moved tokens (and the cameras
# that follow them) differ. Walls without such an opening are unchanged.
#   setsid nohup eval/studies/syncrack3d_v122_junctions_250.sh >> /mnt/nvme_storage/syncrack3d_v1.22.0-junctions_dataset/run250.log 2>&1 < /dev/null & disown
set -u
APP=${APP:-$(cd "$(dirname "$0")/../.." && pwd)}  # this repository
CODE=/mnt/nvme_storage/syncrack3d_v1.22_code
D=/mnt/nvme_storage/syncrack3d_v1.22.0-junctions_dataset
C=$D/train_a
BPY=/home/shiven/anaconda3/envs/infinigen_v2/bin/python
PY=/home/shiven/anaconda3/bin/python
S="nice -n 5 env PYTHONPATH=$APP/eval $PY -m studies.syncrack3d_v116_study"
mkdir -p $C/walls/frames
git -C $CODE rev-parse HEAD > $D/code_commit.txt
git -C $CODE diff > $D/code_diff.patch

if [ ! -f $C/done ]; then
  echo "=== render $(date -Is)"
  for half in 0 1; do
    seeds=$(for s in $(seq 910000 910249); do [ $(((s - 910000) % 2)) = $half ] && echo $s; done | paste -sd,)
    (cd $CODE && for try in 1 2 3; do  # a crashed Blender resumes where it stopped
       $BPY syncrack3d/build_dataset.py --recipe unified --crack-junction-fraction .5 --seeds $seeds \
         --output $C/walls --keep-going --resume >> $C/render_$half.log 2>&1; done) &
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
  echo "$(ls $C/pack/images | wc -l) views, $(ls -d $C/walls/seed_* | wc -l) walls, $(grep -L '"junctions": \[\]' $C/walls/seed_*/scene.json | wc -l) with junction cracks; 910000-910249" > $C/done
  echo "=== done train_a $(cat $C/done) $(date -Is)"
fi

cd $APP
STUDY_TAG=v122jn $S train unet_w250 --walls 250 --arch unet_r34 --seeds 0 1 2 3 4 --iters 4000
STUDY_TAG=v122jn $S eval unet_w250 --splits --real
echo V122JN_250_DONE $(date -Is)
