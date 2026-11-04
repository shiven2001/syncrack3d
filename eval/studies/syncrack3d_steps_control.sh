#!/bin/bash
# Data-vs-steps control: U-Net R34 on all 4,000 v1.17 training walls (23,982 views), training capped
# at 4,000 / 6,000 / 12,000 steps (the v1.17 study scaled steps with views: 23,982). 3 seeds each,
# synthetic val/test + BD3, CUBIT, BuildCrack (protocol v2). Resumable.
#   eval/studies/syncrack3d_steps_control.sh
set -u
APP=${APP:-$(cd "$(dirname "$0")/../.." && pwd)}  # this repository
PY=/home/shiven/anaconda3/bin/python
export STUDY_TAG=v117
S="nice -n 5 env PYTHONPATH=$APP/eval $PY -m studies.syncrack3d_v116_study"
cd $APP
for it in 4000 6000 12000; do
  name=unet_w4000_it$it
  $S train $name --walls 4000 --arch unet_r34 --seeds 0 1 2 --iters $it
  $S eval $name --splits val test --real
  echo "steps $it done $(date -Is)"
done
echo STEPS_CONTROL_DONE
