#!/bin/bash
# T2.1 — Apply PEM to the UA-MT base checkpoint on Pancreas-CT 20%.
# Three seeds, same protocol as PEM on BCP (E=2, mode=full, lr=5e-5).
# Plus the three controlled post-hoc baselines (TS, PL-FT, SC) for the
# same UA-MT base, so the Tab. 1 UA-MT row is structurally identical to
# the BCP row.
#
# Runs only after scripts/run_uamt_pancreas.sh has finished.
set -e
cd /home/tals/Documents/PostHocEM
PY=/home/tals/miniconda3/envs/ns-sam3/bin/python
CKPT=result/uamt_pancreas20/best_model.pth

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: $CKPT not found. Run scripts/run_uamt_pancreas.sh first." >&2
  exit 1
fi

echo "=== PEM on UA-MT Pancreas 20% (3 seeds) ==="
for seed in 2020 42 123; do
    $PY train_posthoc_em.py \
        --checkpoint $CKPT \
        --data_root data/pancreas_h5 \
        --splits_dir splits/pancreas \
        --label_percent 20 \
        --mode full \
        --lr 5e-5 --epochs 2 \
        --patience 100 --min_delta -1.0 \
        --seed $seed \
        --save_dir result/pem_on_uamt_pancreas_${seed} \
        --gpu 0
done

echo "=== Post-hoc baselines on UA-MT Pancreas 20% ==="
for method in ts pl_ft sc; do
    $PY train_baselines.py \
        --method $method \
        --dataset pancreas \
        --checkpoint $CKPT \
        --data_root data/pancreas_h5 \
        --splits_dir splits/pancreas \
        --label_percent 20 \
        --epochs 2 --lr 5e-5 \
        --save_dir result/baseline_${method}_on_uamt_pancreas20 \
        --gpu 0
done

echo "ALL PEM ON UAMT DONE"
