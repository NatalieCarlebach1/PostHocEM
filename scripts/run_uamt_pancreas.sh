#!/bin/bash
# T2.1 — Train UA-MT (Yu et al. 2019) on Pancreas-CT 20% to use as a
# second base SSL method for PEM application.
#
# Target Dice: ~77.26 (BCP paper reports this for UA-MT @ 20%).
# Runtime: ~5 hours on a single RTX 4070 Ti SUPER.
# Run in background:
#   nohup bash scripts/run_uamt_pancreas.sh > /tmp/uamt.log 2>&1 &
set -e
cd /home/tals/Documents/PostHocEM
PY=/home/tals/miniconda3/envs/ns-sam3/bin/python

$PY train_uamt_baseline.py \
    --data_root data/pancreas_h5 \
    --splits_dir splits/pancreas \
    --label_percent 20 \
    --patch_size 96 --batch_size 4 \
    --lr 1e-2 --max_iter 6000 \
    --consistency 0.1 --consistency_rampup 40.0 \
    --threshold_T 8 --U_threshold 0.75 \
    --save_dir result/uamt_pancreas20 \
    --eval_every 200 \
    --seed 2020 --gpu 0

echo "UA-MT BASE TRAINING DONE — checkpoint: result/uamt_pancreas20/best_model.pth"
