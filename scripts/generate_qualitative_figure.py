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


def save_panel(image_slice, mask_slice, color, out_path, label_text=None):
    """Save a single tight panel as PNG. No axes, no padding, no title."""
    h, w = image_slice.T.shape
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=200)
    ax.imshow(image_slice.T, cmap='gray', origin='lower',
              vmin=image_slice.min(), vmax=image_slice.max())
    if mask_slice is not None and mask_slice.any():
        cmap = ListedColormap([(0, 0, 0, 0), color])
        ax.imshow(mask_slice.T, cmap=cmap, origin='lower',
                  alpha=0.5, vmin=0, vmax=1)
        ax.contour(mask_slice.T, levels=[0.5], colors=[color],
                   linewidths=1.4, origin='lower')
    if label_text:
        ax.text(0.02, 0.96, label_text,
                transform=ax.transAxes, color='white', fontsize=14,
                va='top', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.55,
                          edgecolor='none', pad=2))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    out_dir = ROOT / 'paper' / 'figures'
    panels_dir = out_dir / 'panels'
    panels_dir.mkdir(parents=True, exist_ok=True)

    summary = []  # for LaTeX include hints

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

        # Slug for file naming
        slug = cfg['name'].lower().replace(' ', '_').replace('%', 'pct').replace('-', '_')

        # Save the four panels
        save_panel(img_sl, None, None,
                   panels_dir / f'{slug}_image.png')
        save_panel(img_sl, gt_sl, 'lime',
                   panels_dir / f'{slug}_gt.png')
        save_panel(img_sl, b_sl, 'red',
                   panels_dir / f'{slug}_bcp.png',
                   label_text=f'{dice_b*100:.2f}')
        save_panel(img_sl, p_sl, 'cyan',
                   panels_dir / f'{slug}_pem.png',
                   label_text=f'{dice_p*100:.2f} ({delta*100:+.2f})')

        summary.append((cfg['name'], slug, case_name, dice_b, dice_p, delta))
        print(f'  ✓ saved 4 panels for {cfg["name"]} (slug={slug})')

    # Print LaTeX snippet for the user
    print('\n' + '=' * 70)
    print('LaTeX include block (paste into paper/main.tex):')
    print('=' * 70)
    print(r'\begin{figure}[htbp]')
    print(r'\floatconts')
    print(r'  {fig:qualitative}')
    print(r'  {\caption{Qualitative comparison on the test case with the largest PEM improvement for each configuration. Columns: input image, ground truth (green), BCP baseline (red), BCP+PEM (cyan). Per-case Dice is overlaid in each prediction panel; the value in parentheses is the PEM gain over BCP.}}')
    print(r'  {%')
    print(r'    \setlength{\tabcolsep}{1pt}%')
    print(r'    \renewcommand{\arraystretch}{0.2}%')
    print(r'    \begin{tabular}{@{}cccc@{}}')
    print(r'      \footnotesize Image & \footnotesize Ground Truth & \footnotesize BCP & \footnotesize BCP + PEM (ours) \\')
    for name, slug, case, dice_b, dice_p, delta in summary:
        print(rf'      \includegraphics[width=0.23\linewidth]{{figures/panels/{slug}_image}} &')
        print(rf'      \includegraphics[width=0.23\linewidth]{{figures/panels/{slug}_gt}} &')
        print(rf'      \includegraphics[width=0.23\linewidth]{{figures/panels/{slug}_bcp}} &')
        print(rf'      \includegraphics[width=0.23\linewidth]{{figures/panels/{slug}_pem}} \\')
        print(rf'      \multicolumn{{4}}{{c}}{{\footnotesize \textit{{{name}}}}} \\')
    print(r'    \end{tabular}%')
    print(r'  }')
    print(r'\end{figure}')


if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    main()
