#!/bin/bash
# Screen: SynCrack3D v1.18.0 with fixtures mounted on the wall relief (--fixture-mount relief) against v1.18.0
# (the v117lp loose-props walls) at 250 walls (seeds 910000-910249), U-Net R34, 4,000 steps, five seeds,
# zero-shot BD3 / CUBIT / BuildCrack. Every wall moves its fixtures, so all 250 are rendered, two at a time.
#   setsid nohup eval/studies/syncrack3d_fixture_mount_250.sh >> /mnt/nvme_storage/syncrack3d_v1.18.0-fixturemount_dataset/run250.log 2>&1 < /dev/null & disown
set -u
APP=${APP:-$(cd "$(dirname "$0")/../.." && pwd)}  # this repository
CODE=/mnt/nvme_storage/syncrack3d_v1.18-fixturemount_code
D=/mnt/nvme_storage/syncrack3d_v1.18.0-fixturemount_dataset
C=$D/train_a
BPY=/home/shiven/anaconda3/envs/infinigen_v2/bin/python
PY=/home/shiven/anaconda3/bin/python
S="nice -n 5 env PYTHONPATH=$APP/eval $PY -m studies.syncrack3d_v116_study"
mkdir -p $C
git -C $CODE rev-parse HEAD > $D/code_commit.txt
git -C $CODE diff > $D/code_diff.patch

if [ ! -f $C/done ]; then
  echo "=== render $(date -Is)"
  for half in 0 1; do
    seeds=$(seq $((910000 + half)) 2 910249 | paste -sd,)
    (cd $CODE && for try in 1 2 3; do  # a crashed Blender resumes where it stopped
       $BPY syncrack3d/build_dataset.py --recipe unified --fixture-mount relief --seeds $seeds --output $C/walls \
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
STUDY_TAG=v118fm $S train unet_w250 --walls 250 --arch unet_r34 --seeds 0 1 2 3 4 --iters 4000
STUDY_TAG=v118fm $S eval unet_w250 --splits --real
echo FIXTURE_MOUNT_250_DONE $(date -Is)
