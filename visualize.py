"""
Visualization — loss curves + qualitative prediction results
=============================================================
Produces two types of figures:

1. Loss curves: reads TensorBoard event files from result dirs and plots
   train loss + test Dice over epochs for BCP vs DGEM side by side.

2. Qualitative results: loads N random test cases, runs sliding-window
   inference with each model, and saves a grid of:
       [CT image | Ground truth | BCP pred | DGEM pred]
   for three axial/coronal/sagittal slices through the lesion centroid.

Usage:

    # Loss curves only
    python visualize.py losses \
        --result_dirs \
            "BCP baseline:result/bcp_baseline" \
            "DGEM (ours):result/dgem_20p" \
        --out_dir figures/

    # Qualitative predictions
    python visualize.py predictions \
        --data_root  /path/to/pancreas_h5 \
        --test_file  splits/pancreas/test.txt \
        --checkpoints \
            "BCP baseline:result/bcp_baseline/best_model.pth" \
            "DGEM (ours):result/dgem_20p/best_model.pth" \
        --n_cases 4 \
        --out_dir figures/

    # Both at once
    python visualize.py all \
        --result_dirs   "BCP baseline:result/bcp_baseline" "DGEM (ours):result/dgem_20p" \
        --checkpoints   "BCP baseline:result/bcp_baseline/best_model.pth" \
                        "DGEM (ours):result/dgem_20p/best_model.pth" \
        --data_root /path/to/pancreas_h5 \
        --test_file splits/pancreas/test.txt \
        --n_cases 4 \
        --out_dir figures/
"""

import argparse
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py

from networks import VNet
from dataloaders import PancreasDataset
from utils.metrics import sliding_window_inference


# ─────────────────────────────────────────────────────────────────────────────
# Arg parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['losses', 'predictions', 'all'])

    # Loss curves
    p.add_argument('--result_dirs', nargs='+', default=[],
                   metavar='LABEL:PATH',
                   help='TensorBoard result dirs as "Label:path"')

    # Predictions
    p.add_argument('--checkpoints', nargs='+', default=[],
                   metavar='LABEL:PATH',
                   help='Checkpoints as "Label:path/to/best_model.pth"')
    p.add_argument('--data_root',   default=None)
    p.add_argument('--test_file',   default='splits/pancreas/test.txt')
    p.add_argument('--patch_size',  type=int, default=96)
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--n_cases',     type=int, default=4,
                   help='Number of random test cases to visualize')

    p.add_argument('--out_dir',     default='figures')
    p.add_argument('--gpu',         default='0')
    p.add_argument('--seed',        type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard reader (no TF dependency)
# ─────────────────────────────────────────────────────────────────────────────

def read_tb_scalars(event_dir, tags):
    """
    Parse TensorBoard event files and return {tag: [(step, value), ...]}
    Uses tensorboard's EventAccumulator — installed with tensorboardX.
    Falls back to reading the train.log if TB files not found.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(event_dir))
        ea.Reload()
        result = {}
        for tag in tags:
            if tag in ea.Tags().get('scalars', []):
                events = ea.Scalars(tag)
                result[tag] = [(e.step, e.value) for e in events]
        return result
    except Exception as e:
        print(f'  TensorBoard reader failed ({e}), trying log file...')
        return read_log_file(Path(event_dir) / 'train.log')


def read_log_file(log_path):
    """Parse train.log for Dice and loss values as fallback."""
    result = {'test/dice': [], 'train/sup_loss': [], 'train/em_loss': []}
    if not Path(log_path).exists():
        return result
    epoch = 0
    with open(log_path) as f:
        for line in f:
            if 'Epoch [' in line:
                epoch += 1
                for part in line.split():
                    if 'sup=' in part:
                        try:
                            result['train/sup_loss'].append(
                                (epoch, float(part.split('=')[1])))
                        except Exception:
                            pass
            if '[Eval' in line and 'Dice=' in line:
                for part in line.split():
                    if 'Dice=' in part:
                        try:
                            result['test/dice'].append(
                                (epoch, float(part.split('=')[1])))
                        except Exception:
                            pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loss curves
# ─────────────────────────────────────────────────────────────────────────────

COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']


def plot_losses(result_dirs, out_dir):
    """
    Plots test/dice and train/sup_loss curves for all methods.
    result_dirs: list of (label, path) tuples.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Training curves — Pancreas-CT 20% labeled', fontsize=13)

    ax_dice = axes[0]
    ax_loss = axes[1]
    ax_dice.set_title('Test Dice Score')
    ax_dice.set_xlabel('Epoch')
    ax_dice.set_ylabel('Dice')
    ax_loss.set_title('Train Supervised Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')

    for i, (label, path) in enumerate(result_dirs):
        color = COLORS[i % len(COLORS)]
        scalars = read_tb_scalars(path, ['test/dice', 'train/sup_loss',
                                         'train/loss1', 'train/loss2'])

        # Test dice
        dice_data = scalars.get('test/dice', [])
        if dice_data:
            steps, vals = zip(*dice_data)
            ax_dice.plot(steps, vals, color=color, label=label, linewidth=2)
            best = max(vals)
            ax_dice.axhline(best, color=color, linestyle='--', alpha=0.4,
                            linewidth=1)
            ax_dice.annotate(f'{best:.4f}', xy=(steps[-1], best),
                             color=color, fontsize=8, va='bottom')

        # Supervised loss
        loss_key = 'train/sup_loss' if 'train/sup_loss' in scalars else 'train/loss1'
        loss_data = scalars.get(loss_key, [])
        if loss_data:
            steps, vals = zip(*loss_data)
            # Smooth with moving average
            window = max(1, len(vals) // 50)
            smoothed = np.convolve(vals, np.ones(window)/window, mode='valid')
            ax_loss.plot(steps[:len(smoothed)], smoothed,
                         color=color, label=label, linewidth=2)

    ax_dice.legend()
    ax_loss.legend()
    ax_dice.grid(True, alpha=0.3)
    ax_loss.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / 'loss_curves.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

    # Also save EM-specific loss for DGEM
    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.set_title('DGEM — Entropy Loss on Disagreement Voxels')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('EM Loss')
    for i, (label, path) in enumerate(result_dirs):
        scalars = read_tb_scalars(path, ['train/em_loss', 'train/disagree_ratio'])
        em_data = scalars.get('train/em_loss', [])
        if em_data:
            steps, vals = zip(*em_data)
            window = max(1, len(vals) // 50)
            smoothed = np.convolve(vals, np.ones(window)/window, mode='valid')
            ax.plot(steps[:len(smoothed)], smoothed,
                    color=COLORS[i % len(COLORS)], label=label, linewidth=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = out_dir / 'em_loss.png'
    plt.savefig(str(out2), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out2}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Qualitative prediction grid
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, n_classes):
    net = VNet(n_channels=1, n_classes=n_classes,
               normalization='instancenorm', has_dropout=False)
    net = nn.DataParallel(net).cuda()
    state = torch.load(ckpt_path, map_location='cuda')
    if isinstance(state, dict):
        if 'net' in state:
            state = state['net']
        elif 'model_state_dict' in state:
            state = state['model_state_dict']
    net.load_state_dict(state)
    net.eval()
    return net


def get_lesion_centroid(label):
    """Return (x, y, z) centroid of foreground voxels, or volume centre."""
    coords = np.argwhere(label > 0)
    if len(coords) == 0:
        return tuple(s // 2 for s in label.shape)
    return tuple(coords.mean(axis=0).astype(int))


def plot_predictions(checkpoints, data_root, test_file,
                     patch_size, n_classes, n_cases, out_dir, seed):
    """
    For each test case, show axial/coronal/sagittal slices through the
    lesion centroid: [CT | GT | model1_pred | model2_pred | ...]
    """
    random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test case paths
    with open(test_file) as f:
        cases = [l.strip() for l in f if l.strip()]
    random.shuffle(cases)
    cases = cases[:n_cases]

    # Load models
    models = []
    for label, ckpt_path in checkpoints:
        if not Path(ckpt_path).exists():
            print(f'  WARNING: checkpoint not found: {ckpt_path}')
            models.append((label, None))
        else:
            print(f'Loading [{label}] from {ckpt_path}')
            models.append((label, load_model(ckpt_path, n_classes)))

    n_models  = len(models)
    n_cols    = 2 + n_models     # CT | GT | model1 | model2 | ...
    view_names = ['Axial', 'Coronal', 'Sagittal']
    n_rows    = 3                # one per view

    cmap_ct   = 'gray'
    cmap_mask = plt.cm.Reds
    cmap_mask.set_under(alpha=0)   # transparent where mask=0

    for case in cases:
        h5_path = Path(data_root) / case
        with h5py.File(str(h5_path), 'r') as f:
            image = f['image'][:].astype(np.float32)
            label = f['label'][:].astype(np.uint8)

        cx, cy, cz = get_lesion_centroid(label)

        # Run inference for each model
        preds = []
        for lbl, net in models:
            if net is None:
                preds.append(np.zeros_like(label))
            else:
                pred, _ = sliding_window_inference(
                    net, image, patch_size=patch_size,
                    stride_xy=16, stride_z=4, n_classes=n_classes)
                preds.append(pred.astype(np.uint8))

        # Build figure
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3.5 * n_cols, 3.5 * n_rows))
        fig.suptitle(f'Case: {Path(case).stem}', fontsize=11)

        col_titles = ['CT image', 'Ground truth'] + [lbl for lbl, _ in models]

        def show_slice(ax, img_slice, mask_slice, title=''):
            ax.imshow(img_slice, cmap=cmap_ct, vmin=0, vmax=1)
            if mask_slice.sum() > 0:
                ax.imshow(np.ma.masked_less(mask_slice, 0.5),
                          cmap=cmap_mask, alpha=0.55, vmin=0.5, vmax=1)
            ax.set_title(title, fontsize=8)
            ax.axis('off')

        for row, view in enumerate(view_names):
            for col in range(n_cols):
                ax = axes[row, col]

                # Extract slice
                if view == 'Axial':
                    img_sl   = image[:, :, cz]
                    gt_sl    = label[:, :, cz]
                    pred_sls = [p[:, :, cz] for p in preds]
                elif view == 'Coronal':
                    img_sl   = image[:, cy, :]
                    gt_sl    = label[:, cy, :]
                    pred_sls = [p[:, cy, :] for p in preds]
                else:  # Sagittal
                    img_sl   = image[cx, :, :]
                    gt_sl    = label[cx, :, :]
                    pred_sls = [p[cx, :, :] for p in preds]

                if col == 0:
                    show_slice(ax, img_sl, np.zeros_like(img_sl),
                               f'{view} — CT')
                elif col == 1:
                    show_slice(ax, img_sl, gt_sl,
                               f'{view} — GT')
                else:
                    m_idx = col - 2
                    show_slice(ax, img_sl, pred_sls[m_idx],
                               f'{view} — {col_titles[col]}')

        # Legend
        red_patch = mpatches.Patch(color='red', alpha=0.55, label='Pancreas mask')
        fig.legend(handles=[red_patch], loc='lower right', fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = out_dir / f'pred_{Path(case).stem}.png'
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out_path}')

    print(f'\nAll qualitative figures saved to {out_dir}/')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_label_path(entries):
    result = []
    for e in entries:
        if ':' not in e:
            print(f'WARNING: skipping "{e}" — expected "Label:path"')
            continue
        label, path = e.split(':', 1)
        result.append((label, path))
    return result


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    do_losses = args.mode in ('losses', 'all')
    do_preds  = args.mode in ('predictions', 'all')

    if do_losses:
        if not args.result_dirs:
            print('ERROR: --result_dirs required for loss curves')
            sys.exit(1)
        result_dirs = parse_label_path(args.result_dirs)
        print('\n=== Plotting loss curves ===')
        plot_losses(result_dirs, out_dir)

    if do_preds:
        if not args.checkpoints or not args.data_root:
            print('ERROR: --checkpoints and --data_root required for predictions')
            sys.exit(1)
        checkpoints = parse_label_path(args.checkpoints)
        print('\n=== Plotting prediction grids ===')
        plot_predictions(
            checkpoints=checkpoints,
            data_root=args.data_root,
            test_file=args.test_file,
            patch_size=args.patch_size,
            n_classes=args.num_classes,
            n_cases=args.n_cases,
            out_dir=out_dir,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
