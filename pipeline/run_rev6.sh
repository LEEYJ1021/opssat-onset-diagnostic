#!/bin/bash
source ~/venvs/paper2/bin/activate
cd ~/Research/Satellite/"OPSSAT-AD_LEO data"
M=CNN1D,TCN,BiLSTM,BiGRU,TinyTransformer
for S in 1 2 3; do
  python rev6_seq_models.py --neg pseudo --estimators PELT3 --models $M --seed $S --gpu 0 --n-boot 100 > logs_rev/rev6_pseudo_seed$S.log 2>&1
  for f in auc contrasts operating_point; do
    cp revision_outputs/rev6_${f}_pseudo.csv revision_outputs/rev6_${f}_pseudo_seed$S.csv
  done
  python rev6_seq_models.py --neg random --estimators PELT3 --models $M --seed $S --gpu 0 --n-boot 100 > logs_rev/rev6_random_seed$S.log 2>&1
  for f in auc contrasts operating_point; do
    cp revision_outputs/rev6_${f}.csv revision_outputs/rev6_${f}_seed$S.csv
  done
done
echo REV6_DONE > logs_rev/REV6_DONE
