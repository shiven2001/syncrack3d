#!/bin/bash
# Screen: SynCrack3D v1.19.0 with every fixture placed by one rule (structures.sample_structure_plan: a size,
# anchor and hardware pad per kind; every built extent on the wall and 3 cm clear of every other) against v118fc
# (v1.19.0: --fixture-spacing clear) at 250 walls (seeds 910000-910249), U-Net R34, 4,000 steps, five seeds,
# zero-shot BD3 / CUBIT / BuildCrack. Every screen wall's fixture plan changes, so all 250 render, two at a time.
# The code is the v1.19.0 tag with only the planner swapped in ($D/code_diff.patch), so props match v118fc.
#   setsid nohup studies/syncrack3d_fixture_rule_250.sh >> /mnt/nvme_storage/syncrack3d_v1.19.0-fixturerule_dataset/run250.log 2>&1 < /dev/null & disown
set -u
R=/home/shiven/syndefect3d
CODE=/mnt/nvme_storage/syncrack3d_v1.19-fixturerule_code
D=/mnt/nvme_storage/syncrack3d_v1.19.0-fixturerule_dataset
C=$D/train_a
BPY=/home/shiven/anaconda3/envs/infinigen_v2/bin/python
PY=/home/shiven/anaconda3/bin/python
S="nice -n 5 env PYTHONPATH=$R/syncrack3d $PY -m studies.syncrack3d_v116_study"
mkdir -p $C/walls/frames
git -C $CODE rev-parse HEAD > $D/code_commit.txt
git -C $CODE diff > $D/code_diff.patch

if [ ! -f $C/done ]; then
  echo "=== render $(date -Is)"
  for half in 0 1; do
    seeds=$(for s in $(seq 910000 910249); do [ $(((s - 910000) % 2)) = $half ] && echo $s; done | paste -sd,)
    (cd $CODE && for try in 1 2 3; do  # a crashed Blender resumes where it stopped
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

cd $R
STUDY_TAG=v119fr $S train unet_w250 --walls 250 --arch unet_r34 --seeds 0 1 2 3 4 --iters 4000
STUDY_TAG=v119fr $S eval unet_w250 --splits --real
echo FIXTURE_RULE_250_DONE $(date -Is)
