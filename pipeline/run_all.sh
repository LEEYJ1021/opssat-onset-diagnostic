#!/bin/bash
source ~/venvs/paper2/bin/activate
cd ~/Research/Satellite/"OPSSAT-AD_LEO data"
export OMP_NUM_THREADS=4

( python rev9_estimator_agreement.py > logs_rev/rev9.log 2>&1
  python rev8_segment_level.py       > logs_rev/rev8.log 2>&1 ) &
( python rev11_injection.py  > logs_rev/rev11.log 2>&1 ) &
( python rev10_window_scan.py > logs_rev/rev10.log 2>&1 ) &

( for S in 1 2 3; do
    python rev6_seq_models.py --neg pseudo --estimators PELT3 --models CNN1D,TCN,BiLSTM,BiGRU,TinyTransformer \
      --seed $S --gpu 0 --n-boot 100 > logs_rev/rev6_pseudo_seed$S.log 2>&1
    for f in auc contrasts operating_point; do cp revision_outputs/rev6_${f}_pseudo.csv revision_outputs/rev6_${f}_pseudo_seed$S.csv; done
  done ) &

( for S in 1 2; do
    python rev6_seq_models.py --neg random --estimators PELT3 --models CNN1D,TCN,BiLSTM,BiGRU,TinyTransformer \
      --seed $S --gpu 0 --n-boot 100 > logs_rev/rev6_random_seed$S.log 2>&1
    for f in auc contrasts operating_point; do cp revision_outputs/rev6_${f}.csv revision_outputs/rev6_${f}_seed$S.csv; done
  done ) &

wait
echo ALL_DONE > logs_rev/ALL_DONE
