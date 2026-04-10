"""
Unified evaluation script — run on any saved checkpoint.
Reports Dice, Jaccard, HD95, ASD on Pancreas-CT test set.
Also prints a comparison table if multiple checkpoints are given.

Usage (single checkpoint):
    python evaluate.py \
        --checkpoint result/dgem_20p/best_model.pth \
        --data_root  /path/to/pancreas_h5 \
        --test_file  splits/pancreas/test.txt

Usage (comparison table):
    python evaluate.py \
        --data_root /path/to/pancreas_h5 \
        --test_file splits/pancreas/test.txt \
        --compare \
            "BCP baseline:result/bcp_baseline/best_model.pth" \
            "DGEM (ours):result/dgem_20p/best_model.pth" \
            "Post-hoc EM:result/posthoc_em/best_model.pth"
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from networks import VNet
from dataloaders import PancreasDataset
from torch.utils.data import DataLoader
from utils.metrics import evaluate


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',   required=True)
    p.add_argument('--test_file',   default='splits/pancreas/test.txt')
    p.add_argument('--patch_size',  type=int, default=96)
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--gpu',         default='0')

    # Single checkpoint mode
    p.add_argument('--checkpoint',  default=None,
                   help='Single checkpoint .pth to evaluate')

    # Comparison table mode
    p.add_argument('--compare',     nargs='+', default=None,
                   metavar='LABEL:PATH',
                   help='Multiple checkpoints as "Label:path/to/model.pth"')
    return p.parse_args()


def load_model(checkpoint_path, n_classes):
    net = VNet(n_channels=1, n_classes=n_classes,
               normalization='instancenorm', has_dropout=False)
    net = nn.DataParallel(net).cuda()

    state = torch.load(checkpoint_path, map_location='cuda')
    # Handle different save formats
    if isinstance(state, dict):
        if 'net' in state:
            state = state['net']
        elif 'model_state_dict' in state:
            state = state['model_state_dict']
    net.load_state_dict(state)
    net.eval()
    return net


def eval_one(checkpoint_path, test_loader, patch_size, n_classes):
    net = load_model(checkpoint_path, n_classes)
    dice, jc, hd, asd = evaluate(net, test_loader, patch_size=patch_size, n_classes=n_classes)
    return dice, jc, hd, asd


def print_table(rows):
    """rows: list of (label, dice, jc, hd, asd)"""
    header = f"{'Method':<30}  {'Dice':>7}  {'Jc':>7}  {'HD95':>7}  {'ASD':>7}"
    sep    = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)
    for label, dice, jc, hd, asd in rows:
        print(f'{label:<30}  {dice:>7.4f}  {jc:>7.4f}  {hd:>7.2f}  {asd:>7.2f}')
    print(sep + '\n')


def main():
    args = get_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test_ds = PancreasDataset(
        data_root=args.data_root,
        split_file=args.test_file,
        patch_size=args.patch_size,
        augment=False
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    print(f'Test cases: {len(test_ds)}')

    if args.checkpoint:
        # ── Single checkpoint ─────────────────────────────────────────────
        print(f'\nEvaluating: {args.checkpoint}')
        dice, jc, hd, asd = eval_one(
            args.checkpoint, test_loader, args.patch_size, args.num_classes)
        print_table([('checkpoint', dice, jc, hd, asd)])

    elif args.compare:
        # ── Comparison table ──────────────────────────────────────────────
        rows = []
        for entry in args.compare:
            if ':' not in entry:
                print(f'WARNING: skipping "{entry}" — expected "Label:path"')
                continue
            label, path = entry.split(':', 1)
            if not Path(path).exists():
                print(f'WARNING: checkpoint not found: {path}')
                rows.append((label, 0, 0, 999, 999))
                continue
            print(f'Evaluating [{label}]  {path}')
            dice, jc, hd, asd = eval_one(
                path, test_loader, args.patch_size, args.num_classes)
            rows.append((label, dice, jc, hd, asd))
        print_table(rows)

    else:
        print('Provide --checkpoint or --compare. See --help.')
        sys.exit(1)


if __name__ == '__main__':
    main()
