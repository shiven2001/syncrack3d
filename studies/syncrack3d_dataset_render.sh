#!/bin/bash
# Render, export and contract-check a SynCrack3D dataset, chunk by chunk. Resumable.
#   studies/syncrack3d_dataset_render.sh v1.17.0
#   CODE=/mnt/nvme_storage/syncrack3d_v1.25_code studies/syncrack3d_dataset_render.sh v1.25.0   (frozen checkout)
# Splits are seed ranges, so no wall is in two splits; train chunks are nested in seed order.
# Each chunk renders as two Blender processes (odd and even seeds); the walls do not change.
set -u
VERSION=$1
D=/mnt/nvme_storage/syncrack3d_${VERSION}_dataset
BPY=/home/shiven/anaconda3/envs/infinigen_v2/bin/python
CODE=${CODE:-$(cd "$(dirname "$0")/.." && pwd)}
mkdir -p $D
cd $CODE
git rev-parse HEAD > $D/code_commit.txt
git diff > $D/code_diff.patch
for spec in train_a:910000-910499 val:901000-901199 train_b:910500-910999 train_c:911000-911999 \
            test:900000-900299 train_d:912000-913999; do
  name=${spec%%:*}; seeds=${spec#*:}; C=$D/$name
  [ -f $C/done ] && continue
  mkdir -p $C
  echo "=== render $name $seeds $(date -Is)"
  lo=${seeds%-*}; hi=${seeds#*-}
  for half in 0 1; do
    list=$(seq $((lo + half)) 2 $hi | paste -sd,)
    (for try in $(seq 20); do  # Blender segfaults every few hundred walls; --resume picks up where it stopped
       $BPY syncrack3d/build_dataset.py --recipe unified --seeds $list --output $C/walls --keep-going --resume \
         >> $C/render_$half.log 2>&1 && break; done) &
  done
  wait
  built=$(ls $C/walls/seed_*/scene.json | wc -l)  # scene.json is written last
  if [ $built -lt $((hi - lo + 1)) ]; then echo "render $name incomplete: $built of $((hi - lo + 1)) walls"; exit 1; fi
  rm -rf $C/pack
  $BPY tools/prepare_defect_annotated_dataset.py -i $C/walls/frames -o $C/pack --aov-masks auto --instance-masks auto \
    --with-bboxes --max-negative-fraction 1 >> $C/export.log 2>&1
  $BPY tools/check_instance_contract.py --pack $C/pack > $C/contract.txt 2>&1; echo "contract_$name=$?"
  echo "$(ls $C/pack/images | wc -l) views, $built walls" > $C/done
  echo "=== done $name $(cat $C/done) $(date -Is)"
done
echo RENDER_ALL_DONE
