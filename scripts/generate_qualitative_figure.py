"""
Generate the qualitative comparison figure for the paper.

For each of the three benchmark configurations (Pancreas 20%, LA 5%, LA 10%):
  1. Load BCP baseline and BCP+PEM checkpoints
  2. Run sliding-window inference on every test case
  3. Pick the case with the largest PEM improvement
  4. Pick the slice with the most foreground
  5. Render a 4-panel row (Image | GT | BCP | BCP+PEM)

Output: paper/figures/qualitative.png  (3 rows × 4 columns)

Usage:
    python scripts/generate_qualitative_figure.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from medpy.metric.binary import dc

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from networks import VNet
from networks.vnet_bcp_la import VNetBCP_LA
from utils.metrics import sliding_window_inference


# ─── Config: which checkpoints to compare for each row ───────────────────────

CONFIGS = [
    {
        'name'      : 'Pancreas-CT 20%',
        'dataset'   : 'pancreas',
        'baseline'  : ROOT / 'result/bcp_baseline_v2/best_model.pth',
        'pem'       : ROOT / 'result/pem_bcp_full/best_model.pth',
        'data_root' : ROOT / 'data/pancreas_h5',
        'test_split': ROOT / 'splits/pancreas/test.txt',
        'patch'     : (96, 96, 96),
        'stride_xy' : 16,
        'stride_z'  : 4,
        'cmap_overlay': 'Greens',
    },
    {
        'name'      : 'LA 5%',
        'dataset'   : 'la',
        'baseline'  : ROOT / 'result/bcp_pretrained/LA_5.pth',
        'pem'       : ROOT / 'result/pem_la_sweep_5pct_conf_t0.95_lr1e-5/best_model.pth',
        'data_root' : ROOT / 'data/la_h5/2018LA_Seg_Training Set',
        'test_split': ROOT / 'splits/la/test.txt',
        'patch'     : (112, 112, 80),
        'stride_xy' : 18,
        'stride_z'  : 4,
        'cmap_overlay': 'Reds',
    },
    {
        'name'      : 'LA 10%',
        'dataset'   : 'la',
        'baseline'  : ROOT / 'result/bcp_pretrained/LA_10.pth',
        'pem'       : ROOT / 'result/pem_la_sweep_10pct_conf_t0.9_lr5e-6/best_model.pth',
        'data_root' : ROOT / 'data/la_h5/2018LA_Seg_Training Set',
        'test_split': ROOT / 'splits/la/test.txt',
        'patch'     : (112, 112, 80),
        'stride_xy' : 18,
        'stride_z'  : 4,
        'cmap_overlay': 'Blues',
    },
]


# ─── Loading helpers ─────────────────────────────────────────────────────────

def load_pancreas_model(checkpoint_path):
    net = VNet(n_channels=1, n_classes=2,
               normalization='instancenorm', has_dropout=False)
    net = nn.DataParallel(net).cuda()
    state = torch.load(str(checkpoint_path), map_location='cuda')
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    net.load_state_dict(state)
    net.eval()
    return net


def load_la_model(checkpoint_path):
    net = VNetBCP_LA(n_channels=1, n_classes=2)
    net = nn.DataParallel(net).cuda()
    state = torch.load(str(checkpoint_path), map_location='cuda')
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    if not any(k.startswith('module.') for k in state.keys()):
        net.module.load_state_dict(state, strict=False)
    else:
        net.load_state_dict(state)
    net.eval()
    return net


def load_volume(case, dataset, data_root):
    if dataset == 'pancreas':
        path = data_root / case
    else:
        path = data_root / case / 'mri_norm2.h5'
    with h5py.File(str(path), 'r') as f:
        image = f['image'][:].astype(np.float32)
        label = f['label'][:].astype(np.uint8)
    return image, label


# ─── Find the best case per config ───────────────────────────────────────────

def find_best_improvement_case(cfg):
    """Run inference on every test case with both checkpoints, return the case
    with the largest PEM-vs-baseline Dice delta."""
    print(f'\n=== {cfg["name"]} ===')

    if cfg['dataset'] == 'pancreas':
        baseline_net = load_pancreas_model(cfg['baseline'])
        pem_net      = load_pancreas_model(cfg['pem'])
    else:
        baseline_net = load_la_model(cfg['baseline'])
        pem_net      = load_la_model(cfg['pem'])

    with open(cfg['test_split']) as f:
        cases = [l.strip() for l in f if l.strip()]

    deltas = []
    cache = {}
    for case in cases:
        image, label = load_volume(case, cfg['dataset'], cfg['data_root'])
        b_pred, _ = sliding_window_inference(
            baseline_net, image, cfg['patch'],
            cfg['stride_xy'], cfg['stride_z'], n_classes=2)
        p_pred, _ = sliding_window_inference(
            pem_net, image, cfg['patch'],
            cfg['stride_xy'], cfg['stride_z'], n_classes=2)
        b_pred = b_pred.astype(np.uint8)
        p_pred = p_pred.astype(np.uint8)
        if label.sum() == 0 or b_pred.sum() == 0 or p_pred.sum() == 0:
            continue
        dice_b = dc(b_pred, label)
        dice_p = dc(p_pred, label)
        delta  = dice_p - dice_b
        deltas.append((case, dice_b, dice_p, delta))
        cache[case] = (image, label, b_pred, p_pred)
        print(f'  {case}: BCP={dice_b:.4f}  PEM={dice_p:.4f}  Δ={delta:+.4f}')

    deltas.sort(key=lambda x: -x[3])
    best = deltas[0]
    print(f'\n  → BEST: {best[0]}  Δ={best[3]:+.4f}')

    image, label, b_pred, p_pred = cache[best[0]]
    return best, image, label, b_pred, p_pred


# ─── Slice picker ────────────────────────────────────────────────────────────

def pick_slice(label):
    """Pick the axial slice index with the most foreground."""
    fg_per_slice = label.sum(axis=(0, 1))
    return int(np.argmax(fg_per_slice))


# ─── Plotting ────────────────────────────────────────────────────────────────

def overlay(ax, image_slice, mask_slice, color, alpha=0.5, lw=1.5):
    """Show grayscale image with a colored binary mask outline."""
    ax.imshow(image_slice.T, cmap='gray', origin='lower',
              vmin=image_slice.min(), vmax=image_slice.max())
    if mask_slice.any():
        cmap = ListedColormap([(0,0,0,0), color])
        ax.imshow(mask_slice.T, cmap=cmap, origin='lower',
                  alpha=alpha, vmin=0, vmax=1)
        # Add contour for crispness
        ax.contour(mask_slice.T, levels=[0.5], colors=[color],
                   linewidths=lw, origin='lower')
    ax.set_xticks([]); ax.set_yticks([])


def main():
    out_dir = ROOT / 'paper' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(CONFIGS), 4,
                              figsize=(11, 3 * len(CONFIGS)))
    if len(CONFIGS) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Image', 'Ground Truth', 'BCP', 'BCP + PEM (ours)']

    for r, cfg in enumerate(CONFIGS):
        try:
            best, image, label, b_pred, p_pred = find_best_improvement_case(cfg)
        except Exception as e:
            print(f'FAILED for {cfg["name"]}: {e}')
            continue

        case_name, dice_b, dice_p, delta = best
        z = pick_slice(label)
        img_sl = image[:, :, z]
        gt_sl  = label[:, :, z]
        b_sl   = b_pred[:, :, z]
        p_sl   = p_pred[:, :, z]

        # Column 0: image
        axes[r, 0].imshow(img_sl.T, cmap='gray', origin='lower',
                          vmin=img_sl.min(), vmax=img_sl.max())
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(cfg['name'], fontsize=12, fontweight='bold')

        # Column 1: ground truth
        overlay(axes[r, 1], img_sl, gt_sl, color='lime', alpha=0.45)

        # Column 2: BCP prediction
        overlay(axes[r, 2], img_sl, b_sl, color='red', alpha=0.45)
        axes[r, 2].text(0.02, 0.95, f'Dice {dice_b*100:.2f}',
                        transform=axes[r, 2].transAxes, color='white',
                        fontsize=9, va='top',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))

        # Column 3: BCP+PEM prediction
        overlay(axes[r, 3], img_sl, p_sl, color='cyan', alpha=0.45)
        axes[r, 3].text(0.02, 0.95, f'Dice {dice_p*100:.2f} ({delta*100:+.2f})',
                        transform=axes[r, 3].transAxes, color='white',
                        fontsize=9, va='top',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))

    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=12)

    plt.tight_layout()
    out_path = out_dir / 'qualitative.png'
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
    fig.savefig(str(out_dir / 'qualitative.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'\n✓ saved {out_path}')
    print(f'✓ saved {out_dir / "qualitative.pdf"}')


if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    main()
