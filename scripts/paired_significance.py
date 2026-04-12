"""
Paired statistical significance for PEM vs BCP.

For each (BCP, BCP+PEM) checkpoint pair, runs sliding-window inference on
every test case and computes:
  - per-case Dice for both models
  - Δ_i = Dice_PEM - Dice_BCP
  - Wilcoxon signed-rank test (paired, non-parametric)
  - bootstrap 95% CI on mean Δ (10000 resamples, percentile)
  - Cohen's d_z (paired effect size)
  - win/tie/loss

No retraining; one forward pass per checkpoint.
"""

import os, sys, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import h5py
from scipy.stats import wilcoxon
from medpy.metric.binary import dc

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from networks import VNet
from networks.vnet_bcp_la import VNetBCP_LA
from utils.metrics import sliding_window_inference


CONFIGS = [
    {'name': 'Pancreas-CT 20%',
     'kind': 'pancreas',
     'bcp':  'result/bcp_baseline_v2/best_model.pth',
     'methods': {
         'PEM':   'result/pem_seed_pancreas_2020/best_model.pth',
         'PL-FT': 'result/baseline_pl_ft_pancreas20/best_model.pth',
         'SC':    'result/baseline_sc_pancreas20/best_model.pth',
     },
     'data_root': 'data/pancreas_h5',
     'test_split': 'splits/pancreas/test.txt',
     'patch': (96, 96, 96), 'stride_xy': 16, 'stride_z': 4},
    {'name': 'LA 5%',
     'kind': 'la',
     'bcp':  'result/bcp_pretrained/LA_5.pth',
     'methods': {
         'PEM':   'result/pem_seed_la5_2020/best_model.pth',
         'PL-FT': 'result/baseline_pl_ft_la5/best_model.pth',
         'SC':    'result/baseline_sc_la5/best_model.pth',
     },
     'data_root': 'data/la_h5/2018LA_Seg_Training Set',
     'test_split': 'splits/la/test.txt',
     'patch': (112, 112, 80), 'stride_xy': 18, 'stride_z': 4},
    {'name': 'LA 10%',
     'kind': 'la',
     'bcp':  'result/bcp_pretrained/LA_10.pth',
     'methods': {
         'PEM':   'result/pem_seed_la10_2020/best_model.pth',
         'PL-FT': 'result/baseline_pl_ft_la10/best_model.pth',
         'SC':    'result/baseline_sc_la10/best_model.pth',
     },
     'data_root': 'data/la_h5/2018LA_Seg_Training Set',
     'test_split': 'splits/la/test.txt',
     'patch': (112, 112, 80), 'stride_xy': 18, 'stride_z': 4},
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


def per_case_dice(net, cases, kind, data_root, patch, stride_xy, stride_z):
    out = []
    for case in cases:
        if kind == 'pancreas':
            path = ROOT / data_root / case
        else:
            path = ROOT / data_root / case / 'mri_norm2.h5'
        with h5py.File(str(path), 'r') as f:
            image = f['image'][:].astype(np.float32)
            label = f['label'][:].astype(np.uint8)
        pred, _ = sliding_window_inference(
            net, image, patch, stride_xy, stride_z, n_classes=2)
        pred = pred.astype(np.uint8)
        if pred.sum() == 0 or label.sum() == 0:
            out.append(0.0)
        else:
            out.append(dc(pred, label))
    return np.asarray(out)


def bootstrap_ci(deltas, n_boot=10000, alpha=0.05, seed=2020):
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(deltas)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = deltas[idx].mean()
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi


def cohens_dz(deltas):
    return float(deltas.mean() / (deltas.std(ddof=1) + 1e-12))


def main():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    print(f'{"Setting":<16s} {"Method":<8s} {"N":>3s} {"mean Δ":>9s} {"95% CI":>17s} '
          f'{"Wilcoxon p":>12s} {"d_z":>7s} {"W/T/L":>10s}')
    print('-' * 92)

    for cfg in CONFIGS:
        with open(ROOT / cfg['test_split']) as f:
            cases = [l.strip() for l in f if l.strip()]

        loader = load_pancreas if cfg['kind'] == 'pancreas' else load_la

        net_bcp = loader(cfg['bcp'])
        d_bcp = per_case_dice(net_bcp, cases, cfg['kind'], cfg['data_root'],
                              cfg['patch'], cfg['stride_xy'], cfg['stride_z'])
        del net_bcp
        torch.cuda.empty_cache()

        for mname, mckpt in cfg['methods'].items():
            net = loader(mckpt)
            d = per_case_dice(net, cases, cfg['kind'], cfg['data_root'],
                              cfg['patch'], cfg['stride_xy'], cfg['stride_z'])
            del net
            torch.cuda.empty_cache()

            deltas = d - d_bcp
            wins = int((deltas > 1e-4).sum())
            ties = int((np.abs(deltas) <= 1e-4).sum())
            losses = int((deltas < -1e-4).sum())
            try:
                _, p = wilcoxon(d, d_bcp, zero_method='wilcox', alternative='two-sided')
            except ValueError:
                p = float('nan')
            lo, hi = bootstrap_ci(deltas)
            d_z = cohens_dz(deltas)
            n = len(deltas)
            mean_delta = deltas.mean() * 100
            print(f'{cfg["name"]:<16s} {mname:<8s} {n:>3d} {mean_delta:>+8.2f}%  '
                  f'[{lo*100:+5.2f}, {hi*100:+5.2f}]  '
                  f'{p:>10.2e}  {d_z:>+6.2f}  {wins:>3d}/{ties}/{losses}')


if __name__ == '__main__':
    main()
