#!/bin/bash
# T1.3 — BCP retrain on Pancreas-CT 20% with periodic trajectory checkpoints.
#
# Output: result/bcp_trajectory/trajectory/epoch_010.pth, _020.pth, ..., _260.pth
#         (26 checkpoints, one every 10 epochs across the 260-epoch BCP run).
#
# Runtime: ~8 hours on a single RTX 4070 Ti SUPER. Run in background:
#   nohup bash scripts/run_bcp_trajectory.sh > /tmp/bcp_trajectory.log 2>&1 &
set -e
cd /home/tals/Documents/PostHocEM
PY=/home/tals/miniconda3/envs/ns-sam3/bin/python

$PY train_bcp_baseline.py \
    --data_root data/pancreas_h5 \
    --splits_dir splits/pancreas \
    --label_percent 20 \
    --pretrain_epochs 60 \
    --selftrain_epochs 200 \
    --eval_every 10 \
    --save_every 10 \
    --seed 2020 \
    --save_dir result/bcp_trajectory \
    --gpu 0

echo "BCP TRAJECTORY DONE — checkpoints in result/bcp_trajectory/trajectory/"
ls result/bcp_trajectory/trajectory/ | head -30
