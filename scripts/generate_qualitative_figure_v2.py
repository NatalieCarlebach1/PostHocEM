"""
Richer qualitative figure for the SIPAIM paper.

For each of the three benchmark configurations, on the median-improvement
test case, render SIX panels on the most-foreground axial slice:

  Input | Ground truth | BCP | BCP+PEM | Entropy(BCP) | PEM correction

  - Entropy(BCP): per-voxel Shannon entropy of the BCP softmax, magma
    heatmap. Visualizes the thin boundary shell (the rho_f concentration).
  - PEM correction: green = voxels PEM fixes (PEM=GT, BCP!=GT);
    red = voxels PEM breaks (PEM!=GT, BCP=GT). Shows the net effect.

Outputs panels to sipaim/figures/panels/{slug}_{image,gt,bcp,pem,entropy,corr}.png
Reuses the loading/inference/selection logic of generate_qualitative_figure.py.
"""

import os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from generate_qualitative_figure import (
    CONFIGS, load_pancreas_model, load_la_model, load_volume, pick_slice)
from utils.metrics import sliding_window_inference
from medpy.metric.binary import dc


def infer(net, image, cfg):
    label_map, score_map = sliding_window_inference(
        net, image, cfg['patch'], cfg['stride_xy'], cfg['stride_z'], n_classes=2)
    return label_map.astype(np.uint8), score_map  # score_map: (2, W,H,D)


def entropy_from_scores(score_map):
    p = np.clip(score_map, 1e-7, 1.0)
    H = -(p * np.log(p)).sum(axis=0)          # (W,H,D)
    return H / np.log(2.0)                      # bits, in [0,1] for binary


def save_gray_mask(img_sl, mask_sl, color, out, text=None):
    h, w = img_sl.T.shape
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=200)
    ax.imshow(img_sl.T, cmap='gray', origin='lower',
              vmin=img_sl.min(), vmax=img_sl.max())
    if mask_sl is not None and mask_sl.any():
        cmap = ListedColormap([(0, 0, 0, 0), color])
        ax.imshow(mask_sl.T, cmap=cmap, origin='lower', alpha=0.45, vmin=0, vmax=1)
        ax.contour(mask_sl.T, levels=[0.5], colors=[color], linewidths=1.4, origin='lower')
    if text:
        ax.text(0.02, 0.96, text, transform=ax.transAxes, color='white',
                fontsize=14, va='top', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=2))
    ax.set_axis_off(); plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out), dpi=200, bbox_inches='tight', pad_inches=0); plt.close(fig)


def save_entropy(img_sl, ent_sl, out):
    h, w = img_sl.T.shape
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=200)
    ax.imshow(img_sl.T, cmap='gray', origin='lower',
              vmin=img_sl.min(), vmax=img_sl.max())
    m = np.ma.masked_less(ent_sl.T, 0.03)
    ax.imshow(m, cmap='magma', origin='lower', alpha=0.85, vmin=0, vmax=1)
    ax.set_axis_off(); plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out), dpi=200, bbox_inches='tight', pad_inches=0); plt.close(fig)


def save_correction(img_sl, gt_sl, b_sl, p_sl, out):
    gain = ((p_sl == gt_sl) & (b_sl != gt_sl)).astype(np.uint8)   # PEM fixes
    loss = ((p_sl != gt_sl) & (b_sl == gt_sl)).astype(np.uint8)   # PEM breaks
    h, w = img_sl.T.shape
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=200)
    ax.imshow(img_sl.T, cmap='gray', origin='lower',
              vmin=img_sl.min(), vmax=img_sl.max())
    ax.imshow(np.ma.masked_equal(gain.T, 0), cmap=ListedColormap(['#22dd55']),
              origin='lower', alpha=0.9, vmin=0, vmax=1)
    ax.imshow(np.ma.masked_equal(loss.T, 0), cmap=ListedColormap(['#ee2222']),
              origin='lower', alpha=0.9, vmin=0, vmax=1)
    ax.set_axis_off(); plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out), dpi=200, bbox_inches='tight', pad_inches=0); plt.close(fig)


def main():
    panels = ROOT / 'sipaim' / 'figures' / 'panels'
    panels.mkdir(parents=True, exist_ok=True)
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        loader = load_pancreas_model if cfg['dataset'] == 'pancreas' else load_la_model
        bnet, pnet = loader(cfg['baseline']), loader(cfg['pem'])
        cases = [l.strip() for l in open(cfg['test_split']) if l.strip()]
        recs = []
        cache = {}
        for c in cases:
            img, lab = load_volume(c, cfg['dataset'], cfg['data_root'])
            bp, bs = infer(bnet, img, cfg)
            pp, ps = infer(pnet, img, cfg)
            if lab.sum() == 0 or bp.sum() == 0 or pp.sum() == 0:
                continue
            db, dp = dc(bp, lab), dc(pp, lab)
            recs.append((c, db, dp, dp - db)); cache[c] = (img, lab, bp, pp, bs)
            print(f"  {c}: BCP={db:.4f} PEM={dp:.4f} d={dp-db:+.4f}")
        recs.sort(key=lambda x: x[3])
        c, db, dp, d = recs[len(recs) // 2]      # median case (reviewer-defensible)
        print(f"  -> median {c} d={d:+.4f}")
        img, lab, bp, pp, bs = cache[c]
        z = pick_slice(lab)
        img_sl, gt_sl, b_sl, p_sl = img[:, :, z], lab[:, :, z], bp[:, :, z], pp[:, :, z]
        ent_sl = entropy_from_scores(bs)[:, :, z]
        slug = cfg['name'].lower().replace(' ', '_').replace('%', 'pct').replace('-', '_')
        save_gray_mask(img_sl, None, None, panels / f'{slug}_image.png')
        save_gray_mask(img_sl, gt_sl, 'lime', panels / f'{slug}_gt.png')
        save_gray_mask(img_sl, b_sl, 'red', panels / f'{slug}_bcp.png', text=f'{db*100:.2f}')
        save_gray_mask(img_sl, p_sl, 'cyan', panels / f'{slug}_pem.png', text=f'{dp*100:.2f} ({d*100:+.2f})')
        save_entropy(img_sl, ent_sl, panels / f'{slug}_entropy.png')
        save_correction(img_sl, gt_sl, b_sl, p_sl, panels / f'{slug}_corr.png')
        print(f"  saved 6 panels for {slug}")


if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    main()
