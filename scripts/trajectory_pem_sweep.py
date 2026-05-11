"""
T1.3 — the central BMVC experiment.

For each trajectory checkpoint saved by scripts/run_bcp_trajectory.sh,
compute (a) the checkpoint's shell entropy concentration rho_f(0.99) on
the unlabeled training set, (b) the checkpoint's baseline test Dice, and
(c) the post-PEM test Dice under the standard fixed-budget protocol
(E=2, lr=5e-5, seed=2020). Output a CSV with one row per checkpoint and
columns: epoch, rho_f, base_dice, pem_dice, delta_dice, base_hd95,
pem_hd95.

This CSV drives the main BMVC figure: scatter of Delta Dice vs rho_f,
with the supervised-only baseline pinned to the left edge. The expected
shape is a monotone curve that is flat or slightly negative in the
F1 regime (rho_f < 0.85), rising in the well-conditioned middle, and
falling toward zero in the F2 saturation limit (rho_f -> 1).

The script is invoked AFTER scripts/run_bcp_trajectory.sh has finished.
For each trajectory checkpoint it does one rho_f forward pass and one
PEM run, total ~30 minutes for 26 checkpoints. PEM runs reuse the
existing train_posthoc_em.py to keep the protocol identical to the
main table.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from networks import VNet
from utils.metrics import sliding_window_inference


def load_pancreas_model(ckpt_path):
    net = VNet(n_channels=1, n_classes=2,
               normalization='instancenorm', has_dropout=False)
    net = nn.DataParallel(net).cuda()
    state = torch.load(str(ckpt_path), map_location='cuda')
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    net.load_state_dict(state)
    net.eval()
    return net


def compute_rho_f(net, unlabeled_cases, data_root, patch, stride_xy, stride_z,
                  tau=0.99):
    """rho_f(tau) = sum_i H_i * 1[c_i <= tau] / sum_i H_i, averaged over
    volumes via aggregation of numerator and denominator."""
    num = 0.0
    den = 0.0
    for case in unlabeled_cases:
        path = ROOT / data_root / case
        if not path.exists():
            continue
        with h5py.File(str(path), 'r') as f:
            image = f['image'][:].astype(np.float32)
        _, logits = sliding_window_inference(
            net, image, patch, stride_xy, stride_z, n_classes=2,
            return_logits=True) if False else (None, None)
        # Fallback: re-run forward and get softmax+entropy without modifying utils
        from utils.metrics import sliding_window_inference as swi
        pred, soft = swi(net, image, patch, stride_xy, stride_z, n_classes=2)
        # soft is (C, D, H, W) probabilities
        p = np.clip(soft, 1e-7, 1 - 1e-7)
        H = -(p * np.log(p)).sum(axis=0)
        c = p.max(axis=0)
        num += float(H[c <= tau].sum())
        den += float(H.sum())
    return num / max(den, 1e-12)


def evaluate_checkpoint(ckpt_path, test_cases, data_root, patch,
                        stride_xy, stride_z):
    from medpy.metric.binary import dc, hd95, asd
    net = load_pancreas_model(ckpt_path)
    dice_list, hd_list = [], []
    for case in test_cases:
        path = ROOT / data_root / case
        if not path.exists():
            continue
        with h5py.File(str(path), 'r') as f:
            image = f['image'][:].astype(np.float32)
            label = f['label'][:].astype(np.uint8)
        pred, _ = sliding_window_inference(
            net, image, patch, stride_xy, stride_z, n_classes=2)
        pred = pred.astype(np.uint8)
        if pred.sum() == 0 or label.sum() == 0:
            dice_list.append(0.0)
            hd_list.append(np.nan)
        else:
            dice_list.append(dc(pred, label))
            try:
                hd_list.append(hd95(pred, label))
            except Exception:
                hd_list.append(np.nan)
    del net
    torch.cuda.empty_cache()
    return (100.0 * float(np.mean(dice_list)),
            float(np.nanmean(hd_list)))


def run_pem(ckpt_path, out_dir, lr=5e-5, epochs=2, seed=2020):
    """Invoke the existing train_posthoc_em.py on a single checkpoint."""
    cmd = [
        sys.executable, str(ROOT / 'train_posthoc_em.py'),
        '--checkpoint', str(ckpt_path),
        '--data_root', 'data/pancreas_h5',
        '--splits_dir', 'splits/pancreas',
        '--label_percent', '20',
        '--mode', 'full',
        '--lr', f'{lr:g}',
        '--epochs', str(epochs),
        '--patience', '100',
        '--min_delta', '-1.0',
        '--seed', str(seed),
        '--save_dir', str(out_dir),
        '--gpu', '0',
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_dir',
                        default='result/bcp_trajectory/trajectory',
                        help='Where the periodic checkpoints live')
    parser.add_argument('--out_dir',
                        default='result/pem_trajectory_sweep')
    parser.add_argument('--out_csv',
                        default='result/pem_trajectory_sweep/trajectory.csv')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip checkpoints that already have a row in the CSV')
    args = parser.parse_args()

    traj_dir = ROOT / args.trajectory_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(traj_dir.glob('epoch_*.pth'))
    if not ckpts:
        print(f"No checkpoints in {traj_dir}. Run scripts/run_bcp_trajectory.sh first.")
        sys.exit(1)

    # Read unlabeled + test splits
    unl = [l.strip() for l in open(ROOT / 'splits/pancreas/unlabeled.txt') if l.strip()]
    test = [l.strip() for l in open(ROOT / 'splits/pancreas/test.txt') if l.strip()]

    patch = (96, 96, 96)
    stride_xy = 16
    stride_z = 4

    done = set()
    if out_csv.exists() and args.skip_existing:
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                done.add(int(r['epoch']))

    write_header = not out_csv.exists()
    with open(out_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'epoch', 'rho_f', 'base_dice', 'base_hd95',
            'pem_dice', 'pem_hd95', 'delta_dice', 'delta_hd95',
        ])
        if write_header:
            w.writeheader()
            f.flush()

        for ckpt in ckpts:
            epoch = int(ckpt.stem.split('_')[1])
            if epoch in done:
                print(f"skip epoch {epoch} (already in CSV)")
                continue

            print(f"\n=== epoch {epoch} ===")
            # 1) rho_f
            net = load_pancreas_model(ckpt)
            rho = compute_rho_f(net, unl, 'data/pancreas_h5',
                                patch, stride_xy, stride_z)
            del net
            torch.cuda.empty_cache()
            print(f"  rho_f(0.99) = {rho:.4f}")

            # 2) base eval
            base_dice, base_hd = evaluate_checkpoint(
                ckpt, test, 'data/pancreas_h5', patch, stride_xy, stride_z)
            print(f"  base Dice = {base_dice:.2f}  HD95 = {base_hd:.2f}")

            # 3) PEM
            pem_out = out_dir / f'epoch_{epoch:03d}'
            run_pem(ckpt, pem_out, lr=args.lr, epochs=args.epochs,
                    seed=args.seed)

            # 4) eval the PEM-fine-tuned checkpoint
            pem_ckpt = pem_out / 'best_model.pth'
            if not pem_ckpt.exists():
                # fallback: try the final epoch ckpt or last_model.pth
                for cand in ['last_model.pth', 'final.pth',
                             'pem_final.pth', 'best.pth']:
                    if (pem_out / cand).exists():
                        pem_ckpt = pem_out / cand
                        break
            pem_dice, pem_hd = evaluate_checkpoint(
                pem_ckpt, test, 'data/pancreas_h5', patch, stride_xy, stride_z)
            print(f"  PEM Dice  = {pem_dice:.2f}  HD95 = {pem_hd:.2f}")

            w.writerow({
                'epoch': epoch,
                'rho_f': f'{rho:.4f}',
                'base_dice': f'{base_dice:.4f}',
                'base_hd95': f'{base_hd:.4f}',
                'pem_dice': f'{pem_dice:.4f}',
                'pem_hd95': f'{pem_hd:.4f}',
                'delta_dice': f'{pem_dice - base_dice:+.4f}',
                'delta_hd95': f'{pem_hd - base_hd:+.4f}',
            })
            f.flush()

    print(f"\nDone. Trajectory CSV at {out_csv}")


if __name__ == '__main__':
    main()
