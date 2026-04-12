"""
Quantify residual entropy structure in converged checkpoints.

For each checkpoint we compute over all UNLABELED training volumes:
  - fraction of voxels with max_softmax > 0.99 (saturated)
  - per-voxel entropy histogram
  - mean L2 distance from "uncertain" voxels (max_p < 0.99) to the nearest
    foreground/background boundary in the predicted segmentation
  - fraction of total entropy budget contributed by the uncertain shell

Compares:
  - BCP Pancreas-CT 20% checkpoint
  - BCP LA 5% and LA 10% checkpoints
  - Supervised-only Pancreas-CT VNet (DGEM warmup checkpoint, 75.7% Dice)
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from scipy.ndimage import distance_transform_edt as edt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from networks import VNet
from networks.vnet_bcp_la import VNetBCP_LA
from utils.metrics import sliding_window_inference


CONFIGS = [
    {
        'name': 'BCP Pancreas-CT 20%',
        'kind': 'pancreas',
        'ckpt': 'result/bcp_baseline_v2/best_model.pth',
        'data_root': 'data/pancreas_h5',
        'split': 'splits/pancreas/train_unlab_20.txt',
        'patch': (96, 96, 96),
        'stride_xy': 16,
    },
    {
        'name': 'Supervised-only Pancreas-CT (no SSL)',
        'kind': 'pancreas',
        'ckpt': 'result/dgem_20p_v2/best_model.pth',
        'data_root': 'data/pancreas_h5',
        'split': 'splits/pancreas/train_unlab_20.txt',
        'patch': (96, 96, 96),
        'stride_xy': 16,
    },
    {
        'name': 'BCP LA 5%',
        'kind': 'la',
        'ckpt': 'result/bcp_pretrained/LA_5.pth',
        'data_root': 'data/la_h5/2018LA_Seg_Training Set',
        'split': 'splits/la/train_unlab_5.txt',
        'patch': (112, 112, 80),
        'stride_xy': 18,
    },
    {
        'name': 'BCP LA 10%',
        'kind': 'la',
        'ckpt': 'result/bcp_pretrained/LA_10.pth',
        'data_root': 'data/la_h5/2018LA_Seg_Training Set',
        'split': 'splits/la/train_unlab_10.txt',
        'patch': (112, 112, 80),
        'stride_xy': 18,
    },
]


def load_pancreas(ckpt):
    net = VNet(n_channels=1, n_classes=2,
               normalization='instancenorm', has_dropout=False)
    net = nn.DataParallel(net).cuda()
    state = torch.load(str(ROOT / ckpt), map_location='cuda')
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    net.load_state_dict(state)
    net.eval()
    return net


def load_la(ckpt):
    net = VNetBCP_LA(n_channels=1, n_classes=2)
    net = nn.DataParallel(net).cuda()
    state = torch.load(str(ROOT / ckpt), map_location='cuda')
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    if not any(k.startswith('module.') for k in state.keys()):
        net.module.load_state_dict(state, strict=False)
    else:
        net.load_state_dict(state)
    net.eval()
    return net


def load_volume(case, kind, data_root):
    if kind == 'pancreas':
        path = ROOT / data_root / case
    else:
        path = ROOT / data_root / case / 'mri_norm2.h5'
    with h5py.File(str(path), 'r') as f:
        image = f['image'][:].astype(np.float32)
    return image


def analyze(cfg):
    print(f'\n=== {cfg["name"]} ===')
    if cfg['kind'] == 'pancreas':
        net = load_pancreas(cfg['ckpt'])
    else:
        net = load_la(cfg['ckpt'])

    with open(ROOT / cfg['split']) as f:
        cases = [l.strip() for l in f if l.strip()]

    total_vox = 0
    saturated_vox = 0  # max_p > 0.99
    total_entropy = 0.0
    shell_entropy = 0.0  # entropy contributed by max_p <= 0.99 voxels
    shell_distances = []  # per-volume mean shell distance
    n_volumes = 0

    for case in cases[:20]:  # Cap at 20 volumes for speed
        image = load_volume(case, cfg['kind'], cfg['data_root'])
        _, score = sliding_window_inference(
            net, image, cfg['patch'], cfg['stride_xy'], 4, n_classes=2)
        # score: (C, W, H, D)
        probs = score / (score.sum(axis=0, keepdims=True) + 1e-8)
        max_p = probs.max(axis=0)              # (W, H, D)
        argmax = probs.argmax(axis=0).astype(np.uint8)
        H = -(probs * np.log(probs + 1e-8)).sum(axis=0)  # per-voxel entropy

        sat_mask = (max_p > 0.99)
        unc_mask = ~sat_mask

        n_vox = max_p.size
        total_vox += n_vox
        saturated_vox += sat_mask.sum()
        total_entropy += H.sum()
        shell_entropy += H[unc_mask].sum()

        # Distance from uncertain voxels to predicted-foreground boundary
        if unc_mask.any() and argmax.sum() > 0 and (1 - argmax).sum() > 0:
            d_in = edt(argmax == 0)   # distance to fg from bg voxels
            d_out = edt(argmax == 1)  # distance to bg from fg voxels
            d_boundary = np.where(argmax == 1, d_out, d_in)
            mean_d = float(d_boundary[unc_mask].mean())
            shell_distances.append(mean_d)
        n_volumes += 1
        print(f'  {case[:20]:20s}  sat={sat_mask.mean()*100:.1f}%  '
              f'unc_share_of_H={H[unc_mask].sum()/(H.sum()+1e-8)*100:.1f}%  '
              f'mean_d={shell_distances[-1] if shell_distances else float("nan"):.2f}')

    sat_pct = saturated_vox / total_vox * 100
    unc_pct = 100 - sat_pct
    shell_share = shell_entropy / (total_entropy + 1e-8) * 100
    mean_d = float(np.mean(shell_distances)) if shell_distances else float('nan')

    summary = {
        'name': cfg['name'],
        'volumes_analyzed': n_volumes,
        'saturated_pct': float(sat_pct),
        'uncertain_pct': float(unc_pct),
        'shell_entropy_share_pct': float(shell_share),
        'mean_distance_to_boundary_voxels': mean_d,
    }
    print(f'  → SUMMARY: saturated={sat_pct:.1f}%  uncertain={unc_pct:.1f}%  '
          f'shell_owns_{shell_share:.1f}%_of_entropy  mean_d={mean_d:.2f}vox')
    return summary


def main():
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    results = []
    for cfg in CONFIGS:
        try:
            summary = analyze(cfg)
            results.append(summary)
        except Exception as e:
            print(f'  FAILED: {e}')
            results.append({'name': cfg['name'], 'error': str(e)})

    out_path = ROOT / 'paper' / 'figures' / 'entropy_stats.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f'\n✓ saved {out_path}')

    # Pretty table
    print('\n' + '=' * 90)
    print(f'{"Checkpoint":<40s} {"Sat%":>7s} {"Unc%":>7s} {"Shell%H":>9s} {"MeanD":>7s}')
    print('-' * 90)
    for r in results:
        if 'error' in r:
            print(f'{r["name"]:<40s} ERROR')
            continue
        print(f'{r["name"]:<40s} {r["saturated_pct"]:>6.1f}  '
              f'{r["uncertain_pct"]:>6.1f}  {r["shell_entropy_share_pct"]:>8.1f}  '
              f'{r["mean_distance_to_boundary_voxels"]:>6.2f}')


if __name__ == '__main__':
    main()
