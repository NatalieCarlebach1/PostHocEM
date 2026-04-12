#!/bin/bash
# Run PEM with 3 seeds for each main configuration. Reports mean ± std.
set -e
cd /home/tals/Documents/PostHocEM
PY=/home/tals/miniconda3/envs/ns-sam3/bin/python

# Pancreas 20% — full mode, lr=5e-5, fixed E=2
for seed in 2020 42 123; do
    $PY train_posthoc_em.py \
        --checkpoint result/bcp_baseline_v2/best_model.pth \
        --data_root data/pancreas_h5 \
        --splits_dir splits/pancreas \
        --label_percent 20 \
        --mode full \
        --lr 5e-5 \
        --epochs 2 \
        --patience 100 \
        --min_delta -1.0 \
        --seed $seed \
        --save_dir result/pem_seed_pancreas_${seed} \
        --gpu 0
done

# LA 5% — confident t=0.95, lr=1e-5, fixed E=5
for seed in 2020 42 123; do
    $PY train_posthoc_em.py \
        --dataset la \
        --checkpoint result/bcp_pretrained/LA_5.pth \
        --data_root data/la_h5 \
        --la_data_root "data/la_h5/2018LA_Seg_Training Set" \
        --splits_dir splits/la \
        --label_percent 5 \
        --patch_size 112,112,80 \
        --num_classes 2 \
        --mode confident --conf_threshold 0.95 \
        --lr 1e-5 \
        --epochs 5 \
        --patience 100 \
        --min_delta -1.0 \
        --seed $seed \
        --save_dir result/pem_seed_la5_${seed} \
        --gpu 0
done

# LA 10% — confident t=0.9, lr=5e-6, fixed E=5
for seed in 2020 42 123; do
    $PY train_posthoc_em.py \
        --dataset la \
        --checkpoint result/bcp_pretrained/LA_10.pth \
        --data_root data/la_h5 \
        --la_data_root "data/la_h5/2018LA_Seg_Training Set" \
        --splits_dir splits/la \
        --label_percent 10 \
        --patch_size 112,112,80 \
        --num_classes 2 \
        --mode confident --conf_threshold 0.9 \
        --lr 5e-6 \
        --epochs 5 \
        --patience 100 \
        --min_delta -1.0 \
        --seed $seed \
        --save_dir result/pem_seed_la10_${seed} \
        --gpu 0
done

echo "ALL DONE"
