#!/bin/bash
# Screen: SynCrack3D v1.20.0 (combined plaster and stucco, no outdoor dirt) with indoor walls only, against v120nd at
# 250 walls, U-Net R34, 4,000 steps, five seeds, zero-shot BD3 / CUBIT / BuildCrack. v120nd's 22 outdoor (facade)
# walls are replaced by the next 22 indoor walls (910250 onward, every non-facade style draw), so both train on 250
# walls. The 228 indoor walls of 910000-910249 ($OLD_D/unchanged.json) are hard links to the v120nd renders, and the
# 22 new walls render with v120nd's frozen code.
#   setsid nohup studies/syncrack3d_indoor_only_250.sh >> /mnt/nvme_storage/syncrack3d_v1.20.0-indooronly_dataset/run250.log 2>&1 < /dev/null & disown
set -u
R=/home/shiven/syndefect3d
CODE=/mnt/nvme_storage/syncrack3d_v1.20-nodirt_code
OLD_D=/mnt/nvme_storage/syncrack3d_v1.20.0-nodirt_dataset
OLD=$OLD_D/train_a/walls
D=/mnt/nvme_storage/syncrack3d_v1.20.0-indooronly_dataset
C=$D/train_a
BPY=/home/shiven/anaconda3/envs/infinigen_v2/bin/python
PY=/home/shiven/anaconda3/bin/python
S="nice -n 5 env PYTHONPATH=$R/syncrack3d $PY -m studies.syncrack3d_v116_study"
mkdir -p $C/walls/frames
git -C $CODE rev-parse HEAD > $D/code_commit.txt
git -C $CODE diff > $D/code_diff.patch
# the next 22 seeds whose style draw is indoor (not facade)
(cd $R/syncrack3d && $PY -c "
import json
from argparse import Namespace
from syncrack3d.recipes import UNIFIED_MIX, wall_style
a = Namespace(crack_style_mix=UNIFIED_MIX, material_category=None, layout_order='fixtures-first')
new = [s for s in range(910250, 910500) if wall_style(s, a)[0] != 'facade'][:22]
json.dump(dict(linked=json.load(open('$OLD_D/unchanged.json')), rendered=new), open('$D/walls.json', 'w'))")
WALLS="$PY -c \"import json; w = json.load(open('$D/walls.json')); print(' '.join(map(str, w['linked'] + w['rendered'])))\""

if [ ! -f $C/done ]; then
  for s in $($PY -c "import json; print(' '.join(map(str, json.load(open('$D/walls.json'))['linked'])))"); do
    [ -e $C/walls/seed_$s ] || cp -al $OLD/seed_$s $C/walls/seed_$s
    [ -e $C/walls/frames/seed_$s ] || cp -al $OLD/frames/seed_$s $C/walls/frames/seed_$s
  done
  echo "=== linked $(ls -d $C/walls/seed_* | wc -l) walls, render $(date -Is)"
  set -- $($PY -c "import json; print(' '.join(map(str, json.load(open('$D/walls.json'))['rendered'])))")
  for half in 0 1; do
    seeds=$(i=0; for s in "$@"; do [ $((i % 2)) = $half ] && echo $s; i=$((i + 1)); done | paste -sd,)
    [ -n "$seeds" ] && (cd $CODE && for try in 1 2 3; do  # a crashed Blender resumes where it stopped
       $BPY syncrack3d/build_dataset.py --recipe unified --seeds $seeds --output $C/walls \
         --keep-going --resume >> $C/render_$half.log 2>&1; done) &
  done
  wait
  missing=0
  for s in $(eval $WALLS); do [ -f $C/walls/seed_$s/scene.json ] || missing=$((missing + 1)); done
  [ "$missing" = 0 ] || { echo "$missing walls missing, stopping"; exit 1; }
  outdoor=$(grep -l '"environment": "outdoor"' $C/walls/seed_*/scene.json | wc -l)
  [ "$outdoor" = 0 ] || { echo "$outdoor outdoor walls, stopping"; exit 1; }
  (cd $CODE && rm -rf $C/pack
   $BPY tools/prepare_defect_annotated_dataset.py -i $C/walls/frames -o $C/pack --aov-masks auto --instance-masks auto \
     --with-bboxes --max-negative-fraction 1 >> $C/export.log 2>&1
   $BPY tools/check_instance_contract.py --pack $C/pack > $C/contract.txt 2>&1; echo "contract_train_a=$?")
  echo "$(ls $C/pack/images | wc -l) views, $(ls -d $C/walls/seed_* | wc -l) walls, all indoor" > $C/done
  echo "=== done train_a $(cat $C/done) $(date -Is)"
fi

cd $R
STUDY_TAG=v120io $S train unet_w250 --walls 250 --arch unet_r34 --seeds 0 1 2 3 4 --iters 4000
STUDY_TAG=v120io $S eval unet_w250 --splits --real
echo INDOOR_ONLY_250_DONE $(date -Is)
