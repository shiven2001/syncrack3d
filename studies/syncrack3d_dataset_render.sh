#!/bin/bash
# Render, export and contract-check a SynCrack3D dataset, chunk by chunk. Resumable.
#   studies/syncrack3d_dataset_render.sh v1.17.0
# Splits are seed ranges, so no wall is in two splits; train chunks are nested in seed order.
set -u
VERSION=$1
D=/mnt/nvme_storage/syncrack3d_${VERSION}_dataset
BPY=/home/shiven/anaconda3/envs/infinigen_v2/bin/python
mkdir -p $D
cd /home/shiven/syndefect3d/syncrack3d
git rev-parse HEAD > $D/code_commit.txt
for spec in train_a:910000-910499 val:901000-901199 train_b:910500-910999 train_c:911000-911999 \
            test:900000-900299 train_d:912000-913999; do
  name=${spec%%:*}; seeds=${spec#*:}; C=$D/$name
  [ -f $C/done ] && continue
  mkdir -p $C
  echo "=== render $name $seeds $(date -Is)"
  $BPY syncrack3d/build_dataset.py --recipe unified --seeds $seeds --output $C/walls --keep-going --resume >> $C/render.log 2>&1
  rm -rf $C/pack
  $BPY tools/prepare_defect_annotated_dataset.py -i $C/walls/frames -o $C/pack --aov-masks auto --instance-masks auto \
    --with-bboxes --max-negative-fraction 1 >> $C/export.log 2>&1
  $BPY tools/check_instance_contract.py --pack $C/pack > $C/contract.txt 2>&1; echo "contract_$name=$?"
  echo "$(ls $C/pack/images | wc -l) views, $(ls -d $C/walls/seed_* | wc -l) walls" > $C/done
  echo "=== done $name $(cat $C/done) $(date -Is)"
done
echo RENDER_ALL_DONE
