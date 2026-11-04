#!/bin/bash
# Re-score SynCrack3D checkpoints under evaluation protocol v2 (5 px tolerance, 1.5x
# test-time upscaling): BD3 predictions + per-seed scores, CUBIT counts, one BuildCrack
# run over all models, and paired comparisons of each version against the previous one.
#   studies/syncrack3d_eval_v2.sh
set -u
PY=/home/shiven/anaconda3/bin/python
APP=${APP:-$(cd "$(dirname "$0")/.." && pwd)}  # this repository
DT=$APP/outputs/domain_transfer
OUT=$DT/syncrack3d_eval_v2
mkdir -p $OUT/logs
# version label : model folder name (syncrack3d_<name>_bd3_v1) : CUBIT tag
MODELS="v1.2.0:layergate v1.3.0:tilestone v1.4.0:network2 v1.5.0:netstruct v1.6.0-tex:texswap v1.6.0:netorgx"
cd $APP
for mv in $MODELS; do
  v=${mv%%:*}; n=${mv#*:}; M=$DT/syncrack3d_${n}_bd3_v1
  $PY eval/bd3/predict.py --model $M > $OUT/logs/bd3_predict_$n.log 2>&1 || echo "BD3_PREDICT_FAIL $n"
  for s in $M/pred_v2/s*; do $PY eval/bd3/score.py --pred $s --out $OUT/bd3_${n}_$(basename $s).json > /dev/null 2>&1 || echo "BD3_SCORE_FAIL $n $s"; done
  $PY eval/cubit/score.py --model $M --out $OUT/cubit_$n > $OUT/logs/cubit_$n.log 2>&1 || echo "CUBIT_FAIL $n"
  echo "scored $v ($n) $(date -Is)"
done
ARGS=""; for mv in $MODELS; do ARGS="$ARGS --model ${mv%%:*}=$DT/syncrack3d_${mv#*:}_bd3_v1"; done
$PY eval/buildcrack/score.py $ARGS --out $OUT/buildcrack > $OUT/logs/buildcrack.log 2>&1 || echo "BUILDCRACK_FAIL"
prev=""
for mv in $MODELS; do
  n=${mv#*:}
  if [ -n "$prev" ]; then
    $PY eval/cubit/compare.py --a $OUT/cubit_$prev --b $OUT/cubit_$n --out $OUT/cubit_compare_${n}_vs_${prev}.json > /dev/null 2>&1 || echo "CUBIT_COMPARE_FAIL $n"
    (cd $APP && env PYTHONPATH=$APP $PY -c "
from studies import syncrack3d_bd3_compare as c
c.MODELS={'$prev':'syncrack3d_${prev}_bd3_v1','$n':'syncrack3d_${n}_bd3_v1'}
c.OUT=c.DT/'syncrack3d_eval_v2/bd3_compare_${n}_vs_${prev}.json'
c.main()" > $OUT/logs/bd3_compare_$n.log 2>&1) || echo "BD3_COMPARE_FAIL $n"
  fi
  prev=$n
done
$PY eval/cubit/compare.py --a $OUT/cubit_netstruct --b $OUT/cubit_netorgx --out $OUT/cubit_compare_netorgx_vs_netstruct.json > /dev/null 2>&1
(cd $APP && env PYTHONPATH=$APP $PY -c "
from studies import syncrack3d_bd3_compare as c
c.MODELS={'netstruct':'syncrack3d_netstruct_bd3_v1','netorgx':'syncrack3d_netorgx_bd3_v1'}
c.OUT=c.DT/'syncrack3d_eval_v2/bd3_compare_netorgx_vs_netstruct.json'
c.main()" > $OUT/logs/bd3_compare_netorgx_vs_netstruct.log 2>&1)
echo "EVAL_V2_DONE $(date -Is)"
