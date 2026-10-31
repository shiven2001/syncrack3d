#!/bin/bash
# Train one SynCrack3D arm and score it against v1.3.0 and the network/control arms.
#   studies/syncrack3d_arm_eval.sh <name>   (model dir outputs/domain_transfer/syncrack3d_<name>_bd3_v1)
set -u
NAME=$1
PY=/home/shiven/anaconda3/bin/python
R=/home/shiven/syndefect3d
DT=$R/syncrack3d/outputs/domain_transfer
M=$DT/syncrack3d_${NAME}_bd3_v1
LOG=$M/logs
mkdir -p $LOG
cd "$R"
echo "=== train $NAME $(date -Is) ==="
nice -n 5 $PY $M/train_eval.py > $LOG/train.log 2>&1; echo "train_exit=$?"
[ -f $M/runs/s4/last.pt ] || exit 1
for base in tilestone networkctl network; do
  env PYTHONPATH=$R/syncrack3d $PY -c "
from studies import syncrack3d_bd3_compare as c
c.MODELS={'${base}':'syncrack3d_${base}_bd3_v1','$NAME':'syncrack3d_${NAME}_bd3_v1'}
c.OUT=c.DT/'syncrack3d_bd3_compare_${NAME}_vs_${base}.json'
c.main()" > $LOG/bd3_vs_$base.log 2>&1; echo "bd3_vs_${base}_exit=$?"
done
cd $R/syncrack3d
$PY eval/cubit/score.py --model $M --out $DT/cubit_eval_${NAME}_v1 > $LOG/cubit.log 2>&1; echo "cubit_exit=$?"
for base in tilestone networkctl network; do
  $PY eval/cubit/compare.py --a $DT/cubit_eval_${base}_v1 --b $DT/cubit_eval_${NAME}_v1 --out $DT/cubit_compare_${NAME}_vs_${base}.json >> $LOG/cubit_compare.log 2>&1
done
$PY eval/buildcrack/score.py --model v1.3.0=$DT/syncrack3d_tilestone_bd3_v1 --model control=$DT/syncrack3d_networkctl_bd3_v1 \
  --model network=$DT/syncrack3d_network_bd3_v1 --model $NAME=$M --out $DT/buildcrack_${NAME}_v1 > $LOG/buildcrack.log 2>&1
echo "buildcrack_exit=$?"
echo "ARM_DONE $NAME $(date -Is)"
