#!/bin/bash
# T1.5 — TENT (Wang et al., ICLR 2021) head-to-head against PEM.
#
# Same epoch budget / LR / data / seed as PEM, but the trainable parameter
# set is restricted to normalization-layer affine params only. Output rows
# slot into the main results table to isolate "full-parameter PEM" from
# "norm-affine-only post-hoc entropy minimization".
set -e
cd /home/tals/Documents/PostHocEM
PY=/home/tals/miniconda3/envs/ns-sam3/bin/python

echo "=== TENT on Pancreas-CT 20% (BCP base) ==="
$PY train_posthoc_em.py \
    --checkpoint result/bcp_baseline_v2/best_model.pth \
    --data_root data/pancreas_h5 \
    --splits_dir splits/pancreas \
    --label_percent 20 \
    --mode full \
    --tent_mode \
    --lr 5e-5 \
    --epochs 2 \
    --patience 100 --min_delta -1.0 \
    --seed 2020 \
    --save_dir result/tent_pancreas20_s2020 \
    --gpu 0

echo "=== TENT on LA 5% (BCP base) ==="
$PY train_posthoc_em.py \
    --dataset la \
    --checkpoint result/bcp_pretrained/LA_5.pth \
    --data_root data/la_h5 \
    --la_data_root "data/la_h5/2018LA_Seg_Training Set" \
    --splits_dir splits/la \
    --label_percent 5 \
    --patch_size 112,112,80 --num_classes 2 \
    --mode confident --conf_threshold 0.95 \
    --tent_mode \
    --lr 1e-5 \
    --epochs 5 \
    --patience 100 --min_delta -1.0 \
    --seed 2020 \
    --save_dir result/tent_la5_s2020 \
    --gpu 0

echo "=== TENT on LA 10% (BCP base) ==="
$PY train_posthoc_em.py \
    --dataset la \
    --checkpoint result/bcp_pretrained/LA_10.pth \
    --data_root data/la_h5 \
    --la_data_root "data/la_h5/2018LA_Seg_Training Set" \
    --splits_dir splits/la \
    --label_percent 10 \
    --patch_size 112,112,80 --num_classes 2 \
    --mode confident --conf_threshold 0.9 \
    --tent_mode \
    --lr 5e-6 \
    --epochs 5 \
    --patience 100 --min_delta -1.0 \
    --seed 2020 \
    --save_dir result/tent_la10_s2020 \
    --gpu 0

echo "=== TENT DONE on all 3 configurations ==="
