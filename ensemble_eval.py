"""
Ensemble evaluation: average softmax predictions across multiple checkpoints.

Each checkpoint runs sliding-window inference, then we average the score maps
voxel-wise before argmax. This is a free improvement when the models are
complementary (e.g., student vs EMA, different fine-tuning trajectories).

Usage:
    python ensemble_eval.py \
        --checkpoints \
            result/pem_bcp_full/best_model.pth \
            result/pem_ema_full/best_model.pth \
        --data_root data/pancreas_h5 \
        --test_file splits/pancreas/test.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from networks import VNet
from dataloaders import FullVolumeDataset
from utils.metrics import sliding_window_inference, calculate_metric_percase


def load_model(checkpoint_path, num_classes):
    net = VNet(n_channels=1, n_classes=num_classes,
               normalization='instancenorm', has_dropout=False)
    net = nn.DataParallel(net).cuda()
    state = torch.load(checkpoint_path, map_location='cuda')
    if isinstance(state, dict):
        if 'net' in state:
            state = state['net']
        elif 'model_state_dict' in state:
            state = state['model_state_dict']
    net.load_state_dict(state)
    net.eval()
    return net


@torch.no_grad()
def ensemble_predict(models, image, patch_size, stride_xy, stride_z, n_classes,
                     weights=None):
    """Average score maps across models."""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()

    avg_score = None
    for w, model in zip(weights, models):
        _, score = sliding_window_inference(
            model, image, patch_size, stride_xy, stride_z, n_classes)
        if avg_score is None:
            avg_score = w * score
        else:
            avg_score = avg_score + w * score

    label_map = np.argmax(avg_score, axis=0)
    return label_map.astype(np.uint8), avg_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', required=True,
                   help='List of checkpoint .pth files to ensemble')
    p.add_argument('--weights', type=float, nargs='+', default=None,
                   help='Optional per-checkpoint weights (default: uniform)')
    p.add_argument('--data_root', required=True)
    p.add_argument('--test_file', required=True)
    p.add_argument('--patch_size', type=int, default=96)
    p.add_argument('--stride_xy', type=int, default=16)
    p.add_argument('--stride_z',  type=int, default=4)
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--gpu', default='0')
    args = p.parse_args()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(f'Loading {len(args.checkpoints)} models...')
    models = []
    for ckpt in args.checkpoints:
        if not Path(ckpt).exists():
            print(f'  MISSING: {ckpt}'); sys.exit(1)
        models.append(load_model(ckpt, args.num_classes))
        print(f'  OK: {ckpt}')

    if args.weights:
        assert len(args.weights) == len(args.checkpoints), \
            'Number of weights must match number of checkpoints'
        print(f'Weights: {args.weights}')
    else:
        print('Weights: uniform')

    test_ds = FullVolumeDataset(args.data_root, args.test_file)
    print(f'Test cases: {len(test_ds)}')
    print(f'Sliding window: patch={args.patch_size}, stride_xy={args.stride_xy}, stride_z={args.stride_z}')
    print('=' * 60)

    dice_list, jc_list, hd_list, asd_list = [], [], [], []
    for i in range(len(test_ds)):
        image, label, name = test_ds[i]
        pred, _ = ensemble_predict(
            models, image, args.patch_size, args.stride_xy, args.stride_z,
            args.num_classes, args.weights)
        d, jc, hd, asd = calculate_metric_percase(pred, label)
        dice_list.append(d)
        jc_list.append(jc)
        hd_list.append(hd)
        asd_list.append(asd)
        print(f'  {name}: Dice={d:.4f}  Jc={jc:.4f}  HD95={hd:.2f}  ASD={asd:.2f}')

    print('=' * 60)
    print(f'ENSEMBLE RESULTS  ({len(args.checkpoints)} models)')
    print(f'  Dice    : {np.mean(dice_list):.4f}  ({np.std(dice_list):.4f})')
    print(f'  Jaccard : {np.mean(jc_list):.4f}  ({np.std(jc_list):.4f})')
    print(f'  HD95    : {np.mean(hd_list):.2f}  ({np.std(hd_list):.2f})')
    print(f'  ASD     : {np.mean(asd_list):.2f}  ({np.std(asd_list):.2f})')

    return {
        'dice': float(np.mean(dice_list)),
        'jaccard': float(np.mean(jc_list)),
        'hd95': float(np.mean(hd_list)),
        'asd': float(np.mean(asd_list)),
    }


if __name__ == '__main__':
    main()
