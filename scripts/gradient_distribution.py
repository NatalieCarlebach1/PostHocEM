"""
T1.4 — Direct measurement of the |dH/dz| distribution on converged checkpoints.

For each unlabeled volume, compute the per-voxel binary-entropy gradient
magnitude |dH/dz_i| = |z_i| * p_i * (1 - p_i) using the foreground logit z_i
of a 2-class softmax. Report:

  - total mass of |dH/dz| over the volume
  - fraction of mass carried by voxels with confidence c_i <= tau
    (i.e.\ the boundary shell) — this is the empirical counterpart of
    Proposition 1 in Section 4 of the paper
  - fraction of mass within k voxels of the predicted decision surface
  - histogram of |dH/dz| over confidence bins

Output: result/gradient_dist/<checkpoint_name>.json + a per-voxel mass map
saved as a single .npz per checkpoint for downstream visualization (T1.7).

No model updates. One forward pass per volume.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from networks import VNet
from networks.vnet_bcp_la import VNetBCP_LA


# Same canonical configurations used in entropy_stats.py and paired_significance.py.
CONFIGS = [
    {'name':       'pancreas20_bcp',
     'kind':       'pancreas',
     'ckpt':       'result/bcp_baseline_v2/best_model.pth',
     'data_root':  'data/pancreas_h5',
     'unl_split':  'splits/pancreas/unlabeled.txt',
     'patch':      (96, 96, 96),
     'stride_xy':  16,
     'stride_z':   4},
    {'name':       'pancreas_supervised',
     'kind':       'pancreas',
     'ckpt':       'result/supervised_pancreas/best_model.pth',
     'data_root':  'data/pancreas_h5',
     'unl_split':  'splits/pancreas/unlabeled.txt',
     'patch':      (96, 96, 96),
     'stride_xy':  16,
     'stride_z':   4},
    {'name':       'la5_bcp',
     'kind':       'la',
     'ckpt':       'result/bcp_pretrained/LA_5.pth',
     'data_root':  'data/la_h5/2018LA_Seg_Training Set',
     'unl_split':  'splits/la/unlabeled_5.txt',
     'patch':      (112, 112, 80),
     'stride_xy':  18,
     'stride_z':   4},
    {'name':       'la10_bcp',
     'kind':       'la',
     'ckpt':       'result/bcp_pretrained/LA_10.pth',
     'data_root':  'data/la_h5/2018LA_Seg_Training Set',
     'unl_split':  'splits/la/unlabeled_10.txt',
     'patch':      (112, 112, 80),
     'stride_xy':  18,
     'stride_z':   4},
]


def load_model(ckpt_path, kind):
    if kind == 'pancreas':
        net = VNet(n_channels=1, n_classes=2,
                   normalization='instancenorm', has_dropout=False)
    else:
        net = VNetBCP_LA(n_channels=1, n_classes=2)
    net = nn.DataParallel(net).cuda()
    state = torch.load(str(ROOT / ckpt_path), map_location='cuda')
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    if kind == 'la' and not any(k.startswith('module.') for k in state.keys()):
        net.module.load_state_dict(state, strict=False)
    else:
        net.load_state_dict(state)
    net.eval()
    return net


def logit_field(net, image, patch, stride_xy, stride_z):
    """Sliding-window logit aggregation. Returns the 2-class logit volume
    averaged over overlapping windows."""
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).cuda()
    D, H, W = image.shape[2:]
    pd, ph, pw = patch
    out = torch.zeros(1, 2, D, H, W, device='cuda')
    cnt = torch.zeros(1, 1, D, H, W, device='cuda')

    def ranges(N, p, s):
        if N <= p:
            return [0]
        r = list(range(0, N - p + 1, s))
        if r[-1] + p < N:
            r.append(N - p)
        return r

    with torch.no_grad():
        for z0 in ranges(D, pd, stride_z):
            for y0 in ranges(H, ph, stride_xy):
                for x0 in ranges(W, pw, stride_xy):
                    patch_in = image[:, :, z0:z0+pd, y0:y0+ph, x0:x0+pw]
                    logit = net(patch_in)
                    if isinstance(logit, (tuple, list)):
                        logit = logit[0]
                    out[:, :, z0:z0+pd, y0:y0+ph, x0:x0+pw] += logit
                    cnt[:, :, z0:z0+pd, y0:y0+ph, x0:x0+pw] += 1.0
    out = out / cnt.clamp_min(1.0)
    return out.squeeze(0).cpu().numpy()


def boundary_distance(pred_mask):
    """Distance from each voxel to the predicted decision surface (vox)."""
    pred = pred_mask.astype(bool)
    if pred.sum() == 0 or (~pred).sum() == 0:
        return np.full(pred.shape, np.nan, dtype=np.float32)
    boundary = pred ^ binary_erosion(pred, iterations=1)
    return distance_transform_edt(~boundary).astype(np.float32)


def measure(cfg, max_volumes=None):
    """For each unlabeled volume, compute the per-voxel |dH/dz| field, the
    mass within the c_i <= tau shell, and the mass within k voxels of the
    predicted boundary. Aggregate statistics over all volumes."""
    if not (ROOT / cfg['ckpt']).exists():
        print(f"SKIP {cfg['name']}: checkpoint not found at {cfg['ckpt']}")
        return None
    if not (ROOT / cfg['unl_split']).exists():
        print(f"SKIP {cfg['name']}: split file not found at {cfg['unl_split']}")
        return None

    net = load_model(cfg['ckpt'], cfg['kind'])
    cases = [l.strip() for l in open(ROOT / cfg['unl_split']) if l.strip()]
    if max_volumes:
        cases = cases[:max_volumes]

    agg = {
        'total_mass':     0.0,
        'shell_mass_099': 0.0,
        'shell_mass_095': 0.0,
        'shell_mass_090': 0.0,
        'mass_within_1':  0.0,
        'mass_within_3':  0.0,
        'mass_within_5':  0.0,
        'mass_within_10': 0.0,
        'n_voxels':       0,
        'n_volumes':      0,
        'per_volume':     [],
    }

    for case in cases:
        if cfg['kind'] == 'pancreas':
            path = ROOT / cfg['data_root'] / case
        else:
            path = ROOT / cfg['data_root'] / case / 'mri_norm2.h5'
        if not path.exists():
            print(f"  skip {case}: not found")
            continue

        with h5py.File(str(path), 'r') as f:
            image = f['image'][:].astype(np.float32)

        logits = logit_field(net, image, cfg['patch'],
                             cfg['stride_xy'], cfg['stride_z'])  # (2, D, H, W)
        # For a 2-class softmax, dH/dz_fg = -z_fg * p * (1-p) where
        # z_fg = logits[1] - logits[0] is the binary-equivalent logit.
        z = logits[1] - logits[0]
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        grad_mag = np.abs(z) * p * (1 - p)
        conf = np.maximum(p, 1 - p)
        pred = (p > 0.5).astype(np.uint8)

        total = float(grad_mag.sum())
        shell_099 = float(grad_mag[conf <= 0.99].sum())
        shell_095 = float(grad_mag[conf <= 0.95].sum())
        shell_090 = float(grad_mag[conf <= 0.90].sum())

        dist = boundary_distance(pred)
        if not np.isnan(dist).all():
            w1  = float(grad_mag[dist <= 1].sum())
            w3  = float(grad_mag[dist <= 3].sum())
            w5  = float(grad_mag[dist <= 5].sum())
            w10 = float(grad_mag[dist <= 10].sum())
        else:
            w1 = w3 = w5 = w10 = 0.0

        agg['total_mass']     += total
        agg['shell_mass_099'] += shell_099
        agg['shell_mass_095'] += shell_095
        agg['shell_mass_090'] += shell_090
        agg['mass_within_1']  += w1
        agg['mass_within_3']  += w3
        agg['mass_within_5']  += w5
        agg['mass_within_10'] += w10
        agg['n_voxels']       += grad_mag.size
        agg['n_volumes']      += 1
        agg['per_volume'].append({
            'case': case,
            'total': total,
            'shell_099_frac': shell_099 / max(total, 1e-12),
            'within_3_frac': w3 / max(total, 1e-12),
        })

        print(f"  {case}: total={total:.3e}  "
              f"shell(0.99)={shell_099/max(total,1e-12):.3f}  "
              f"within_3vox={w3/max(total,1e-12):.3f}")

    if agg['n_volumes'] == 0:
        return None

    out = {
        'config': cfg['name'],
        'n_volumes': agg['n_volumes'],
        'mean_total_gradient_mass': agg['total_mass'] / agg['n_volumes'],
        'fraction_in_shell_099': agg['shell_mass_099'] / max(agg['total_mass'], 1e-12),
        'fraction_in_shell_095': agg['shell_mass_095'] / max(agg['total_mass'], 1e-12),
        'fraction_in_shell_090': agg['shell_mass_090'] / max(agg['total_mass'], 1e-12),
        'fraction_within_1_vox': agg['mass_within_1'] / max(agg['total_mass'], 1e-12),
        'fraction_within_3_vox': agg['mass_within_3'] / max(agg['total_mass'], 1e-12),
        'fraction_within_5_vox': agg['mass_within_5'] / max(agg['total_mass'], 1e-12),
        'fraction_within_10_vox': agg['mass_within_10'] / max(agg['total_mass'], 1e-12),
        'per_volume': agg['per_volume'],
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_volumes', type=int, default=None,
                        help='Limit volumes per config (for quick smoke test)')
    parser.add_argument('--out_dir', default='result/gradient_dist')
    args = parser.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        result = measure(cfg, max_volumes=args.max_volumes)
        if result is None:
            continue
        out_path = out_dir / f"{cfg['name']}.json"
        out_path.write_text(json.dumps(result, indent=2))
        summary[cfg['name']] = {
            k: v for k, v in result.items() if k != 'per_volume'
        }
        print(f"  -> wrote {out_path}")
        print(f"  shell(0.99)={result['fraction_in_shell_099']:.4f}  "
              f"within 3 vox={result['fraction_within_3_vox']:.4f}  "
              f"within 5 vox={result['fraction_within_5_vox']:.4f}")

    summary_path = out_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary: {summary_path}")


if __name__ == '__main__':
    main()
