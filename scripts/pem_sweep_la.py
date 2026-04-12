"""
PEM hyperparameter sweep on LA (Left Atrium MRI) at 5% and 10% labels.

For each (fraction, lr, mode) configuration, runs train_posthoc_em.py with
--dataset la and collects the best Dice. Skips already-completed runs.
At the end, prints a summary table and writes summary.csv.

LA needs much gentler LRs than Pancreas because:
  - Higher baseline (89.4% at 10%) → less residual entropy headroom
  - BatchNorm makes the model more brittle to large updates
  - The PEM lr=5e-5 (Pancreas default) collapses LA in 1 epoch

Usage:
    python scripts/pem_sweep_la.py                  # full LA sweep
    python scripts/pem_sweep_la.py --fractions 10   # only 10% fraction
    python scripts/pem_sweep_la.py --dry_run        # print commands, don't run
"""

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


# ─── LA-specific defaults ────────────────────────────────────────────────────

DEFAULT_FRACTIONS = [5, 10]

# Very gentle LRs — LA collapses with anything above 1e-5
DEFAULT_LRS = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]

# Confidence thresholds for the `confident` mode
DEFAULT_THRESHOLDS = [0.9, 0.95, 0.99, 0.999]

DEFAULT_EPOCHS = 10
DEFAULT_PATIENCE = 5

# LA constants
LA_PATCH_SIZE = '112,112,80'
LA_STRIDE_XY  = 18
LA_NUM_CLASSES = 2
LA_DATA_ROOT  = 'data/la_h5'
LA_INNER_DIR  = 'data/la_h5/2018LA_Seg_Training Set'
LA_SPLITS_DIR = 'splits/la'


# ─── Helpers ─────────────────────────────────────────────────────────────────

def run_name(fraction, lr, mode, threshold):
    lr_str = f'{lr:.0e}'.replace('-0', '-')
    if mode == 'confident':
        return f'pem_la_sweep_{fraction}pct_conf_t{threshold}_lr{lr_str}'
    return f'pem_la_sweep_{fraction}pct_full_lr{lr_str}'


def parse_metrics_csv(csv_path):
    """Return (best_dice, best_delta, best_epoch, baseline_dice) or None."""
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 1:
        return None
    baseline = float(rows[0]['dice'])
    best_dice = baseline
    best_delta = 0.0
    best_epoch = 0
    for r in rows[1:]:
        dice = float(r['dice'])
        if dice > best_dice:
            best_dice = dice
            best_delta = dice - baseline
            best_epoch = int(r['epoch'])
    return best_dice, best_delta, best_epoch, baseline


def get_checkpoint_path(root, fraction):
    """Resolve the BCP LA checkpoint for a given label fraction."""
    return root / 'result' / 'bcp_pretrained' / f'LA_{fraction}.pth'


def run_one(fraction, lr, mode, threshold, epochs, patience, root, dry_run=False):
    name = run_name(fraction, lr, mode, threshold)
    save_dir = root / 'result' / name
    csv_path = save_dir / 'metrics.csv'

    existing = parse_metrics_csv(csv_path)
    if existing is not None:
        return name, existing, 'cached'

    checkpoint = get_checkpoint_path(root, fraction)
    if not checkpoint.exists():
        return name, None, f'missing checkpoint: {checkpoint}'

    cmd = [
        'python', 'train_posthoc_em.py',
        '--dataset', 'la',
        '--checkpoint', str(checkpoint),
        '--data_root', LA_DATA_ROOT,
        '--la_data_root', LA_INNER_DIR,
        '--splits_dir', LA_SPLITS_DIR,
        '--label_percent', str(fraction),
        '--patch_size', LA_PATCH_SIZE,
        '--stride_xy', str(LA_STRIDE_XY),
        '--num_classes', str(LA_NUM_CLASSES),
        '--mode', mode,
        '--lr', str(lr),
        '--epochs', str(epochs),
        '--patience', str(patience),
        '--save_dir', str(save_dir),
        '--gpu', '0',
    ]
    if mode == 'confident':
        cmd.extend(['--conf_threshold', str(threshold)])

    if dry_run:
        print(' '.join(cmd))
        return name, None, 'dry_run'

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if res.returncode != 0:
        tail = '\n'.join(res.stderr.splitlines()[-5:]) if res.stderr else 'no stderr'
        print(f'  FAILED:\n{tail}')
        return name, None, f'failed (exit {res.returncode})'

    metrics = parse_metrics_csv(csv_path)
    return name, metrics, f'ok ({elapsed:.0f}s)'


def _write_summary(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'fraction', 'lr', 'mode', 'threshold',
            'baseline_dice', 'best_dice', 'best_delta', 'best_epoch',
            'name', 'status',
        ])
        writer.writeheader()
        writer.writerows(rows)


# ─── Main sweep ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fractions', type=int, nargs='+', default=DEFAULT_FRACTIONS)
    p.add_argument('--lrs',       type=float, nargs='+', default=DEFAULT_LRS)
    p.add_argument('--thresholds', type=float, nargs='*', default=DEFAULT_THRESHOLDS,
                   help='Confidence thresholds. Pass --thresholds (no values) to skip threshold sweep.')
    p.add_argument('--epochs',    type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--patience',  type=int, default=DEFAULT_PATIENCE)
    p.add_argument('--include_full', action='store_true', default=True)
    p.add_argument('--no_full',   dest='include_full', action='store_false')
    p.add_argument('--root',      default='/home/tals/Documents/PostHocEM')
    p.add_argument('--dry_run',   action='store_true')
    args = p.parse_args()

    root = Path(args.root)
    summary_path = root / 'result' / 'pem_la_sweep_summary.csv'
    log_path = root / 'result' / 'pem_la_sweep.log'
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Sweep logger ────────────────────────────────────────────────────────
    log_fh = open(log_path, 'a', buffering=1)

    def log(msg):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'{ts}  {msg}'
        print(line)
        log_fh.write(line + '\n')

    # Build (mode, threshold) tuples
    mode_threshold_pairs = []
    if args.include_full:
        mode_threshold_pairs.append(('full', None))
    for t in args.thresholds:
        mode_threshold_pairs.append(('confident', t))

    n_total = len(args.fractions) * len(args.lrs) * len(mode_threshold_pairs)
    est_minutes = n_total * 1.5  # LA runs are faster than Pancreas
    log('=' * 80)
    log(f'LA SWEEP START — log file: {log_path}')
    log(f'Total runs to execute: {n_total}  (estimated ~{est_minutes/60:.1f} hours)')
    log(f'  Fractions:  {args.fractions}')
    log(f'  LRs:        {args.lrs}')
    log(f'  Thresholds: {args.thresholds}')
    log(f'  Modes:      {mode_threshold_pairs}')
    log(f'  Epochs:     {args.epochs}')
    log(f'  Patience:   {args.patience}')
    log('=' * 80)

    rows = []
    counter = 0
    t_start = time.time()

    for fraction in args.fractions:
        ckpt = get_checkpoint_path(root, fraction)
        if not ckpt.exists():
            log(f'[Fraction {fraction}%] checkpoint missing: {ckpt}')
            counter += len(args.lrs) * len(mode_threshold_pairs)
            continue

        for lr in args.lrs:
            for mode, threshold in mode_threshold_pairs:
                counter += 1
                elapsed = (time.time() - t_start) / 60
                t_label = f' t={threshold}' if threshold else ''
                log(f'[{counter}/{n_total}] ({elapsed:.1f}m)  '
                    f'frac={fraction}%  lr={lr}  mode={mode}{t_label}')

                name, metrics, status = run_one(
                    fraction, lr, mode, threshold,
                    args.epochs, args.patience, root, args.dry_run)

                if metrics is not None:
                    best_dice, best_delta, best_epoch, baseline = metrics
                    marker = '★' if best_delta > 0.001 else ('-' if best_delta < -0.001 else ' ')
                    log(f'  → {status}  {marker} baseline={baseline:.4f}  '
                        f'best={best_dice:.4f}  delta={best_delta:+.4f}  ep={best_epoch}  '
                        f'name={name}')
                    rows.append({
                        'fraction': fraction, 'lr': lr, 'mode': mode,
                        'threshold': threshold or '',
                        'baseline_dice': baseline,
                        'best_dice': best_dice,
                        'best_delta': best_delta,
                        'best_epoch': best_epoch,
                        'name': name, 'status': status,
                    })
                else:
                    log(f'  → {status}  name={name}')
                    rows.append({
                        'fraction': fraction, 'lr': lr, 'mode': mode,
                        'threshold': threshold or '',
                        'baseline_dice': '', 'best_dice': '', 'best_delta': '',
                        'best_epoch': '', 'name': name, 'status': status,
                    })

                _write_summary(summary_path, rows)

    # ─── Final summary ──────────────────────────────────────────────────────
    log('=' * 80)
    log('BEST CONFIGURATION PER LABEL FRACTION (LA)')
    log('=' * 80)
    log(f'{"Frac":>5}  {"Baseline":>9}  {"Best":>9}  {"Delta":>9}  '
        f'{"LR":>10}  {"Mode":<22}  {"Ep":>3}')
    log('-' * 80)
    for fraction in args.fractions:
        frac_rows = [r for r in rows
                     if r['fraction'] == fraction
                     and r['best_delta'] != '']
        if not frac_rows:
            log(f'{fraction:>4}%   no successful runs')
            continue
        best = max(frac_rows, key=lambda r: r['best_delta'])
        baseline = best['baseline_dice']
        mode_str = best['mode']
        if best['threshold']:
            mode_str += f" (t={best['threshold']})"
        log(f"{fraction:>4}%  {baseline:>9.4f}  {best['best_dice']:>9.4f}  "
            f"{best['best_delta']:>+9.4f}  {best['lr']:>10.0e}  "
            f"{mode_str:<22}  {best['best_epoch']:>3}")

    log(f'Full CSV results written to: {summary_path}')
    log(f'Sweep log written to:        {log_path}')
    log_fh.close()


if __name__ == '__main__':
    main()
