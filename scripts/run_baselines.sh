#!/bin/bash
# Run the 3 post-hoc baselines (TS, PL-FT, SC) on all 3 configurations.
# Same fixed-budget protocol as PEM (E=2 Pancreas, E=5 LA), single seed.
set -e
cd /home/tals/Documents/PostHocEM
PY=/home/tals/miniconda3/envs/ns-sam3/bin/python

for method in ts pl_ft sc; do
  echo "=== ${method^^} on Pancreas-CT 20% ==="
  $PY train_baselines.py \
      --method $method \
      --dataset pancreas \
      --checkpoint result/bcp_baseline_v2/best_model.pth \
      --data_root data/pancreas_h5 \
      --splits_dir splits/pancreas \
      --label_percent 20 \
      --epochs 2 --lr 5e-5 \
      --save_dir result/baseline_${method}_pancreas20 \
      --gpu 0

  echo "=== ${method^^} on LA 5% ==="
  $PY train_baselines.py \
      --method $method \
      --dataset la \
      --checkpoint result/bcp_pretrained/LA_5.pth \
      --data_root data/la_h5 \
      --la_data_root "data/la_h5/2018LA_Seg_Training Set" \
      --splits_dir splits/la \
      --label_percent 5 \
      --patch_size 112,112,80 --num_classes 2 \
      --epochs 5 --lr 1e-5 \
      --save_dir result/baseline_${method}_la5 \
      --gpu 0

  echo "=== ${method^^} on LA 10% ==="
  $PY train_baselines.py \
      --method $method \
      --dataset la \
      --checkpoint result/bcp_pretrained/LA_10.pth \
      --data_root data/la_h5 \
      --la_data_root "data/la_h5/2018LA_Seg_Training Set" \
      --splits_dir splits/la \
      --label_percent 10 \
      --patch_size 112,112,80 --num_classes 2 \
      --epochs 5 --lr 5e-6 \
      --save_dir result/baseline_${method}_la10 \
      --gpu 0
done

echo "ALL BASELINES DONE"
