#!/bin/bash
# Scaling study + final three-architecture runs on a rendered SynCrack3D dataset. Resumable.
#   STUDY_TAG=v117 studies/syncrack3d_dataset_study.sh v1.17.0
# Scaling: U-Net + ResNet-34, 2 seeds, 250/500/1000/2000(/4000) train walls. Stop at 2000 when
# 1000 -> 2000 adds < 0.005 F1 and < 0.01 IoU on SYNTHETIC val (real data never steer it).
# Final: U-Net R34, DeepLabV3+ R50, SegFormer-B2, 3 seeds, synthetic val/test, frozen evalset v2,
# BD3, CUBIT-InSeg and BuildCrack (protocol v2: 5 px, 1.5x upscale).
set -u
VERSION=$1
D=/mnt/nvme_storage/syncrack3d_${VERSION}_dataset
R=/home/shiven/syndefect3d
DT=$R/syncrack3d/outputs/domain_transfer
PY=/home/shiven/anaconda3/bin/python
S="nice -n 5 env PYTHONPATH=$R/syncrack3d $PY -m studies.syncrack3d_v116_study"
cd $R
wait_chunk() { until [ -f $D/$1/done ]; do sleep 120; done; echo "chunk $1 ready $(date -Is)"; }
point() {
  $S train unet_w$1 --walls $1 --arch unet_r34 --seeds 0 1 && $S eval unet_w$1 --splits val --real
  echo "point $1 done $(date -Is)"
}
gain() {
  $PY - "$1" "$2" <<PYEOF
import json, sys
v = [json.load(open(f"$DT/syncrack3d_${STUDY_TAG}_unet_w{w}/results/synth_val.json"))["summary"]["all_cracked"] for w in sys.argv[1:]]
df, di = v[1]["f1"][0] - v[0]["f1"][0], v[1]["crack_iou"][0] - v[0]["crack_iou"][0]
print(f"{df:.4f} {di:.4f}", "continue" if df >= 0.005 or di >= 0.01 else "stop")
PYEOF
}
wait_chunk train_a; wait_chunk val
point 250; point 500
wait_chunk train_b; point 1000
wait_chunk train_c; point 2000
DECISION=$(gain 1000 2000); echo "decision 1000->2000: $DECISION" | tee -a $D/scaling_decision.txt
if [[ $DECISION == *continue ]]; then
  wait_chunk train_d; point 4000
  echo "decision 2000->4000: $(gain 2000 4000)" | tee -a $D/scaling_decision.txt
  N=4000
else
  wait_chunk test
  for p in $(ps -eo pid,args | grep -E "syncrack3d_dataset_render.sh $VERSION|build_dataset.py --recipe unified --seeds 912000" | grep -v grep | awk '{print $1}'); do kill $p; done
  echo "stopped train_d rendering" | tee -a $D/scaling_decision.txt
  N=2000
fi
echo "N=$N" | tee -a $D/scaling_decision.txt
wait_chunk test
for arch in unet_r34 deeplabv3plus_r50 segformer_b2; do
  name=$([ $arch = unet_r34 ] && echo unet_w$N || echo ${arch}_w$N)
  $S train $name --walls $N --arch $arch --seeds 0 1 2
  $S eval $name --splits val test --real
  M=$DT/syncrack3d_${STUDY_TAG}_$name
  [ -f $M/results/synth_evalset_v2.json ] || (cd $R/syncrack3d && $PY eval/synth/score.py --set $R/syncrack3d/outputs/syncrack3d_evalset_v2 \
      --model $M --save $M/results/synth_evalset_v2.json > /dev/null 2>&1)
  echo "final $name done $(date -Is)"
done
echo STUDY_ALL_DONE
