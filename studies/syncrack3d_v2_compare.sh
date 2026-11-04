#!/bin/bash
# Is SynCrack3D v2.0.0 as good as the tested v1.17.0? Same 1,000 training walls (seeds 910000-910999),
# U-Net R34, 4,000 steps, seeds 0-2, zero-shot BD3 / CUBIT / BuildCrack (protocol v2) for both versions.
# v2.0.0 renders from its own checkout. Resumable.
#   setsid nohup studies/syncrack3d_v2_compare.sh >> /mnt/nvme_storage/syncrack3d_v2.0.0_dataset/compare.log 2>&1 < /dev/null & disown
set -u
APP=${APP:-$(cd "$(dirname "$0")/.." && pwd)}  # this repository
CODE=/mnt/nvme_storage/syncrack3d_v2.0.0_code
D=/mnt/nvme_storage/syncrack3d_v2.0.0_dataset
BPY=/home/shiven/anaconda3/envs/infinigen_v2/bin/python
PY=/home/shiven/anaconda3/bin/python
S="nice -n 5 env PYTHONPATH=$APP $PY -m studies.syncrack3d_v116_study"
mkdir -p $D
git -C $CODE rev-parse HEAD > $D/code_commit.txt

# v1.17 reference at the same size and budget, trained while v2.0.0 renders
(cd $APP && STUDY_TAG=v117 $S train unet_w1000_it4000 --walls 1000 --arch unet_r34 --seeds 0 1 2 --iters 4000 \
  && STUDY_TAG=v117 $S eval unet_w1000_it4000 --splits --real && echo "v117 reference done $(date -Is)") &
REF=$!

for spec in train_a:910000-910499 train_b:910500-910999; do
  name=${spec%%:*}; seeds=${spec#*:}; C=$D/$name
  [ -f $C/done ] && continue
  mkdir -p $C
  echo "=== render $name $seeds $(date -Is)"
  (cd $CODE && $BPY syncrack3d/build_dataset.py --seeds $seeds --output $C/walls --keep-going --resume >> $C/render.log 2>&1
   rm -rf $C/pack
   $BPY tools/prepare_defect_annotated_dataset.py -i $C/walls/frames -o $C/pack --aov-masks auto --instance-masks auto \
     --with-bboxes --max-negative-fraction 1 >> $C/export.log 2>&1
   $BPY tools/check_instance_contract.py --pack $C/pack > $C/contract.txt 2>&1; echo "contract_$name=$?")
  echo "$(ls $C/pack/images | wc -l) views, $(ls -d $C/walls/seed_* | wc -l) walls" > $C/done
  echo "=== done $name $(cat $C/done) $(date -Is)"
done

wait $REF
cd $APP
STUDY_TAG=v200 $S train unet_w1000_it4000 --walls 1000 --arch unet_r34 --seeds 0 1 2 --iters 4000
STUDY_TAG=v200 $S eval unet_w1000_it4000 --splits --real
echo V2_COMPARE_DONE $(date -Is)
