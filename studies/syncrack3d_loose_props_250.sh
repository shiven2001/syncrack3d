#!/bin/bash
# Screen: SynCrack3D v1.17.0 with loose props (--prop-placement loose) against v1.17.0 at 250 walls
# (seeds 910000-910249), U-Net R34, 4,000 steps, five seeds each, zero-shot BD3 / CUBIT / BuildCrack.
# Walls with no prop under either rule are hard links to the v1.17.0 renders (syncrack3d_loose_props.sh
# linked and rendered them up to 910200 before this screen replaced the 1,000-wall run). Resumable.
#   setsid nohup studies/syncrack3d_loose_props_250.sh >> /mnt/nvme_storage/syncrack3d_v1.17.0-looseprops_dataset/run250.log 2>&1 < /dev/null & disown
set -u
R=/home/shiven/syndefect3d
CODE=/mnt/nvme_storage/syncrack3d_v1.17-looseprops_code
D=/mnt/nvme_storage/syncrack3d_v1.17.0-looseprops_dataset
C=$D/train_a
BPY=/home/shiven/anaconda3/envs/infinigen_v2/bin/python
PY=/home/shiven/anaconda3/bin/python
S="nice -n 5 env PYTHONPATH=$R/syncrack3d $PY -m studies.syncrack3d_v116_study"
SEEDS="0 1 2 3 4"
cd $R

# v1.17.0 reference: seeds 0-1 exist at 250 walls, add 2-4
(STUDY_TAG=v117 $S train unet_w250 --walls 250 --arch unet_r34 --seeds $SEEDS --iters 4000 \
  && STUDY_TAG=v117 $S eval unet_w250 --splits --real && echo "v117 w250 reference done $(date -Is)") &
REF=$!

if [ ! -f $C/done ]; then
  redo=$($PY -c "
import json, os
print(','.join(str(s) for s in range(910000, 910250) if not os.path.exists('$C/walls/seed_%d/scene.json' % s)))")
  echo "=== rendering ${redo:-nothing} $(date -Is)"
  [ -n "$redo" ] && (cd $CODE && $BPY syncrack3d/build_dataset.py --recipe unified --prop-placement loose --seeds $redo \
    --output $C/walls --keep-going --resume >> $C/render.log 2>&1)
  missing=$($PY -c "
import os
print(sum(not os.path.exists('$C/walls/seed_%d/scene.json' % s) for s in range(910000, 910250)))")
  [ "$missing" = 0 ] || { echo "$missing of the first 250 walls missing, stopping"; exit 1; }
  (cd $CODE && rm -rf $C/pack
   $BPY tools/prepare_defect_annotated_dataset.py -i $C/walls/frames -o $C/pack --aov-masks auto --instance-masks auto \
     --with-bboxes --max-negative-fraction 1 >> $C/export.log 2>&1
   $BPY tools/check_instance_contract.py --pack $C/pack > $C/contract.txt 2>&1; echo "contract_train_a=$?")
  rm -rf $C/labels
  echo "$(ls $C/pack/images | wc -l) views, $(ls -d $C/walls/seed_* | wc -l) walls; complete for 910000-910249 only" > $C/done
  echo "=== done train_a $(cat $C/done) $(date -Is)"
fi

STUDY_TAG=v117lp $S train unet_w250 --walls 250 --arch unet_r34 --seeds $SEEDS --iters 4000
STUDY_TAG=v117lp $S eval unet_w250 --splits --real
wait $REF
echo LOOSE_PROPS_250_DONE $(date -Is)
